"""
LLaDA (Large Language Diffusion with Masking) Model Implementation
Based on llama2.c with ByT5's byte-level encoding
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ModelArgs:
    dim: int = 4096           # Transformer 維度
    n_layers: int = 32        # 層數
    n_heads: int = 32         # 注意力頭數
    n_kv_heads: Optional[int] = None  # 若為 None 則與 n_heads 相同
    vocab_size: int = 256     # Byte-level 編碼，因此 vocab_size 為 256
    multiple_of: int = 256    # FFN 隱藏層大小必須是此數字的倍數
    hidden_dim: Optional[int] = None  # 若為 None，預設為 4 倍模型維度；此處將在 FFN 中調整為 2/3
    norm_eps: float = 1e-5    # LayerNorm epsilon
    max_seq_len: int = 2048   # 最大序列長度
    dropout: float = 0.0      # dropout 比率

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    shape = [d if i in (1, ndim - 1) else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # 保存原始 shape
    orig_shape_q = xq.shape
    orig_shape_k = xk.shape
    # 將最後一維重塑為兩個部分 (實部, 虛部)
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # 構造 rotary embedding 的複數形式，並增加 batch 與 head 的維度進行廣播：
    freqs = torch.view_as_complex(torch.stack([freqs_cos, freqs_sin], dim=-1))
    # 使 freqs 形狀變為 (1, seqlen, 1, head_dim//2)
    freqs = freqs.unsqueeze(0).unsqueeze(2)
    # 應用 rotary embedding
    xq_rot = xq_complex * freqs
    xk_rot = xk_complex * freqs
    # 轉回實數並恢復原始 shape
    xq_out = torch.view_as_real(xq_rot).reshape(orig_shape_q)
    xk_out = torch.view_as_real(xk_rot).reshape(orig_shape_k)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bs, seqlen, n_kv_heads, head_dim = x.shape
    return (x.unsqueeze(-2).expand(-1, -1, -1, n_rep, -1).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim))

class MaskedAttention(nn.Module):
    """
    使用 diffusion-based 隨機遮罩的多頭注意力層
    不使用因果遮罩，並以隨機比例 t 對所有 token 進行獨立遮罩
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        # 計算 Q, K, V
        xq = self.wq(x).view(bsz, seqlen, self.n_heads_q, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        # 應用 rotary embedding
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos[:seqlen], freqs_sin[:seqlen])
        # 對 KV 重複展開，以配合多查詢注意力
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)
        # 計算注意力分數 (注意：不使用因果遮罩)
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # diffusion-based 隨機遮罩：
        if self.training:
            # 對每個樣本從 U(0,1) 中獨立抽樣一個遮罩概率 t
            t = torch.rand(bsz, device=x.device)  # shape: (bsz,)
            key_len = scores.size(-1)
            # 為每個樣本生成 shape (key_len,) 的遮罩 (True 表示該位置被遮罩)
            mask_prob = t.unsqueeze(1).expand(bsz, key_len)
            mask_tensor = torch.rand(bsz, key_len, device=x.device) < mask_prob
            # 確保至少保留第一個 token 未被遮罩，避免全遮罩導致 NaN
            mask_tensor[:, 0] = False
            # 擴展 mask 至 scores 的 shape: (bsz, 1, 1, key_len)
            scores = scores.masked_fill(mask_tensor.unsqueeze(1).unsqueeze(1), float('-inf'))
        scores = F.softmax(scores, dim=-1)
        scores = self.attn_dropout(scores)
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class FeedForward(nn.Module):
    """
    位置式前饋網路，採用 SwiGLU 激活
    將隱藏層維度調整為原先的 2/3，再按 multiple_of 對齊
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = args.hidden_dim or 4 * args.dim
        # 調整隱藏層維度為 2/3
        hidden_dim = int(2 * hidden_dim / 3)
        if args.multiple_of:
            hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class DiffusionTransformerBlock(nn.Module):
    """融合 diffusion masking 的 Transformer Block"""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attention = MaskedAttention(args)
        self.feed_forward = FeedForward(args)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class LLaDA(nn.Module):
    """
    LLaDA 模型：基於 diffusion 的遮罩預測
    不使用 autoregressive 的因果遮罩，且利用隨機遮罩的 forward process 進行預訓練
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        assert args.vocab_size == 256, "LLaDA 採用 byte-level 編碼，vocab_size 必須為 256"
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layers = nn.ModuleList([DiffusionTransformerBlock(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 預先計算 rotary embeddings 頻率
        freqs_cos, freqs_sin = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.apply(self._init_weights)
        # 為某些殘差投影應用縮放初始化
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)
        logits = self.output(h)
        if targets is not None:
            # 損失僅計算 masked token 的預測（假設 targets 中未 masked 的位置為特定標記，例如 -100)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            self.last_loss = loss
        else:
            # 推理時僅返回最後一個 token 的 logits
            logits = self.output(h[:, -1:, :])
            self.last_loss = None
        return logits

    @torch.inference_mode()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        N = sum(p.numel() for p in self.parameters())
        L, H, Q, T = self.args.n_layers, self.args.n_heads, self.args.dim // self.args.n_heads, self.args.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter / dt
        flops_promised = 312e12
        return flops_achieved / flops_promised
