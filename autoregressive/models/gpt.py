# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
from dataclasses import dataclass
from typing import Optional, List
import logging


import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from utils.drop_path import DropPath
from autoregressive.models.utils.tokens import TokenType
from autoregressive.models.generate import sample
from autoregressive.models.utils.adam import get_autoregressive_structure

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    HAS_NEW_SDP = True
except ImportError:
    HAS_NEW_SDP = False
    import torch.backends.cuda

from autoregressive.models.utils.visulization import *

import numpy as np
import math

def split_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02

    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    adam_block_size: int = 8
    subpass_len: int = None  # divide each pass into several subpasses, each subpass has subpass_len tokens
    subpass_num: int = None  # divide each pass into several subpasses, each pass has subpass_num subpasses
    logger: Optional[logging.Logger] = None
    is_random: bool = False  # whether to use random attention mask for training and generation
    sample_folder_dir: str = "samples"
    pre_token_choose: str = "close_min"  # how to choose the pre-token for each token, options: close_min, close_max, close_unattach_min, close_unattach_max
    freqs_cis_reorder_shceme: str = None # how to reorder the freqs_cis for position embedding
    interlacing_type: str = "adam"  # interlacing type, options: adam, spin_adam
    target_aware_emb: bool = False  # whether to use target-aware embedding
    is_adaLN: bool = False  # whether to use adaLN-Zero in reorder blocks
    use_conditioned_blocks: bool = False  # whether to use ConditionedEnhancementBlock instead of TransformerBlock
    num_inputreorder_modules: int = None  # number of modules in layers_inputreorder, must be a divisor of (len(ori_masked_coords)-1), None means use all passes
    use_pass_aware_adaLN: bool = False  # whether to use pass-aware adaLN in ConditionedEnhancementBlock (add pass_idx as additional condition)
    use_class_aware_adaLN: bool = False  # whether to use class-aware adaLN in ConditionedEnhancementBlock (add class label as additional condition)
    use_kq_norm: bool = False  # whether to apply RMSNorm to queries/keys (KQ-norm)
    class_pass_emb_dim: int = None  # if not None, the dimension of class and pass embedding for adaLN modulation, default to model dim


#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


#################################################################################
#                      Embedding Layers for Pass Index                          #
#################################################################################
class PassEmbedder(nn.Module):
    """
    Embeds pass index into vector representations for pass-aware conditioning.
    """
    def __init__(self, num_passes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_passes, hidden_size)
        self.num_passes = num_passes

    def forward(self, pass_idx: torch.Tensor):
        """
        Args:
            pass_idx: (B,) tensor of pass indices, or scalar tensor
        Returns:
            embeddings: (B, 1, C) tensor
        """
        # Handle scalar or 1D tensor
        if pass_idx.dim() == 0:
            pass_idx = pass_idx.unsqueeze(0)
        embeddings = self.embedding_table(pass_idx).unsqueeze(1)  # (B, 1, C)
        return embeddings


#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))
        self.__flops__ = 0
        self.__macs__ = 0

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        self.use_kq_norm = config.use_kq_norm
        if self.use_kq_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        # if dist.get_rank() == 0:
        #     print("-"*50)
        #     print(f"dim: {self.dim}, Attention: n_head: {self.n_head}, n_kv_head: {self.n_kv_head}, head_dim: {self.head_dim}, total_kv_dim: {total_kv_dim}")
        #     print("-"*50)

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None, 
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        if self.use_kq_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None: # different from RandAR
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        # if dist.get_rank() == 0:
        #     print(f"xq shape: {xq.shape}, is_contiguous: {xq.is_contiguous()}")
        #     print(f"keys shape: {keys.shape}, is_contiguous: {keys.is_contiguous()}")
        #     # print(f"values shape: {values.shape}, is_contiguous: {values.is_contiguous()}")
        #     if mask is not None:
        #         print(f"mask shape: {mask.shape}, is_contiguous: {mask.is_contiguous()}")

        output = F.scaled_dot_product_attention(
            xq, keys, values, 
            attn_mask=mask, 
            is_causal=True if mask is None else False, # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)            
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulates the input tensor x using shift and scale tensors for adaptive normalization.
    Supports both token-wise and sequence-wise conditioning.
    """
    # x: (B, L, C)
    # shift/scale can be either (B, 1, C) or (B, L, C)
    return x * (1 + scale) + shift

class ConditionedEnhancementBlock(nn.Module):
    """
    一个用于 Pass 间特征增强的 Block，使用 AdaLN 机制注入条件信息 c。
    该 Block 假定其输入 x 是需要被优化的 Pass 特征 (如 h_curr_pass)。
    """
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config) # FFN
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps) 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # AdaLN 调制层：用于生成 Attention 和 FFN 的 shift, scale, gate
        # 我们需要 3 * hidden_size for Attention (shift, scale, gate)
        #              3 * hidden_size for FFN (shift, scale, gate)
        # 总共 6 * hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), # SiLU 激活函数
            nn.Linear(config.class_pass_emb_dim if (config.class_pass_emb_dim != config.dim and config.class_pass_emb_dim != None) else config.dim,
                    6 * config.dim, bias=True)
        )

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, # c 是条件特征 (例如: 条件 token C 的聚合表示)
        freqs_cis: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        # c can be either (B, 1, C) for sequence-wise or (B, L, C) for token-wise conditioning
        # shift_msa, scale_msa, gate_msa: (B, 1, C) or (B, L, C)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # 2. Modulated Attention Block
        attn_input = modulate(self.attention_norm(x), shift_msa, scale_msa)

        attn_output = self.attention(attn_input, freqs_cis, input_pos, mask)

        # Apply gate and residual connection
        h = x + self.drop_path(gate_msa * attn_output)

        # 3. Modulated FeedForward Block
        ffn_input = modulate(self.ffn_norm(h), shift_mlp, scale_mlp)
        # compute ffn output
        ffn_output = self.feed_forward(ffn_input)

        # Apply gate and residual connection
        out = h + self.drop_path(gate_mlp * ffn_output)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None):
        h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        self.adam_block_size = config.adam_block_size
        ######################################################
        self.logger = config.logger
        self.subpass_len = config.subpass_len
        self.subpass_num = config.subpass_num
        self.is_random = config.is_random # randomly reorder the tokens within each pass
        self.sample_folder_dir = config.sample_folder_dir
        self.pre_token_choose = config.pre_token_choose
        self.freqs_cis_reorder_shceme = config.freqs_cis_reorder_shceme
        self.interlacing_type = config.interlacing_type
        self.target_aware_emb = config.target_aware_emb  # whether to use target-aware embedding
        self.is_adaLN = config.is_adaLN
        self.use_conditioned_blocks = config.use_conditioned_blocks  # whether to use ConditionedEnhancementBlock
        self.num_inputreorder_modules = config.num_inputreorder_modules  # number of modules in layers_inputreorder
        self.use_class_aware_adaLN = config.use_class_aware_adaLN  # whether to use pass-aware adaLN
        self.use_pass_aware_adaLN = config.use_pass_aware_adaLN  # whether to use pass-aware adaLN
        self.class_pass_emb_dim = config.class_pass_emb_dim if config.class_pass_emb_dim is not None else config.dim  # dimension of class and pass embedding for adaLN modulation
        ######################################################
        if self.model_type == 'c2i':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        elif self.model_type == 't2i':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
        else:
            raise Exception("please check model type")
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        # auxiliary autoregressive training/generation structure
        width = int(config.block_size ** 0.5)
        height = width
        #####################################################################
        self.logger.info(f"self.interlacing_type: {self.interlacing_type}")
        self.auto_regr_struct = get_autoregressive_structure(width=width, height=height, 
                                                             base_block_size=config.adam_block_size, 
                                                             cond_len=config.cls_token_num, 
                                                             logger=self.logger, subpass_len=self.subpass_len, subpass_num=self.subpass_num,
                                                             pre_token_choose=self.pre_token_choose,
                                                             freqs_cis_reorder_shceme=self.freqs_cis_reorder_shceme,
                                                             interlacing_type=self.interlacing_type)
        self.ori_masked_coords = self.auto_regr_struct.ori_masked_coords
        self.first_pass_token_num = self.ori_masked_coords[0].shape[0]

        # Pass embedding for pass-aware adaLN
        if self.use_pass_aware_adaLN:
            # total_passes = len(self.ori_masked_coords)
            total_passes = len(self.auto_regr_struct.decoded_masked_coords) # 7 levels - 10 passes
            self.pass_embedding = PassEmbedder(num_passes=total_passes, hidden_size=self.class_pass_emb_dim)
            self.logger.info(f"Created PassEmbedder for {total_passes} passes, and the embedding dimension is {self.class_pass_emb_dim}")
        
        # Class embedding for class-aware adaLN
        if self.use_class_aware_adaLN and self.class_pass_emb_dim != config.dim:
            self.cls_adaLN_emb = LabelEmbedder(config.num_classes, self.class_pass_emb_dim, config.class_dropout_prob)
            self.logger.info(f"Created ClassEmbedder in AdaLN for {config.num_classes} classes, and the embedding dimension is {self.class_pass_emb_dim}")


        # Precompute KNN plan
        if self.pre_token_choose == 'knn':
            self.knn_idx_names = []
            self.knn_w_names = []
            knn_idxs, knn_ws = self.precompute_knn_plan_all_passes(self.ori_masked_coords)
            for i, (knn_idx, knn_w) in enumerate(zip(knn_idxs, knn_ws)):
                self.register_buffer(f'knn_idx_pass{i}', knn_idx)
                self.register_buffer(f'knn_w_pass{i}', knn_w)
                self.knn_idx_names.append(f'knn_idx_pass{i}')
                self.knn_w_names.append(f'knn_w_pass{i}')

        self.learnable_pos_embedding = nn.Parameter(torch.randn(config.dim))

        # 2-layer MLP with GELU
        if self.target_aware_emb:
            # set intermediate dimension: d_model * 4 or d_model * 2
            d_mid = config.dim * 4 
            
            self.pos_emb_mlp = nn.Sequential(
                nn.Linear(config.dim, d_mid),
                nn.GELU(),
                nn.Linear(d_mid, config.dim)
            )
            self._init_mlp_weights()
        
        # transformer blocks for input token sequence reordering
        if self.pre_token_choose == 'transformer_choose':
            self.layers_inputreorder = torch.nn.ModuleList()
            total_passes = len(self.ori_masked_coords) - 1  # exclude first pass

            # Determine number of modules
            self.logger.info(f"expected num_inputreorder_modules: {self.num_inputreorder_modules}, total_passes: {total_passes}")
            if self.num_inputreorder_modules is None:
                self.num_inputreorder_modules = total_passes
            else:
                # Validate that num_inputreorder_modules is a divisor of total_passes
                assert total_passes % self.num_inputreorder_modules == 0, \
                    f"num_inputreorder_modules ({self.num_inputreorder_modules}) must be a divisor of total_passes ({total_passes})"

            # Calculate how many passes each module handles
            self.passes_per_module = total_passes // self.num_inputreorder_modules
            self.logger.info(f"Creating {self.num_inputreorder_modules} input reorder modules, each handling {self.passes_per_module} passes")

            # Create the modules
            for module_i in range(self.num_inputreorder_modules):
                if self.is_adaLN:
                    self.layers_inputreorder.append(ConditionedEnhancementBlock(config, drop_path=0.0))
                else:
                    self.layers_inputreorder.append(TransformerBlock(config, drop_path=0.0))

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            if self.use_conditioned_blocks:
                # 主 transformer layers 可以选择是否使用 pass_aware_adaLN
                self.layers.append(ConditionedEnhancementBlock(config, dpr[layer_id]))
            else:
                self.layers.append(TransformerBlock(config, dpr[layer_id]))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 2d rotary pos embedding
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)
        # Sinusoidal pos embedding for next-token prediction
        SinusoidalPosEmb_2d = get_2d_sincos_pos_embed(embed_dim=self.config.dim, grid_size=grid_size, cls_token_num=self.cls_token_num)
        SinusoidalPosEmb_2d = torch.tensor(SinusoidalPosEmb_2d)
        assemble_SinusoidalPosEmb_2d = self.auto_regr_struct.assemble_sinusoidal_positional_embedding(SinusoidalPosEmb_2d)
        self.register_buffer('assemble_SinusoidalPosEmb_2d', assemble_SinusoidalPosEmb_2d)
        # if dist.get_rank() == 0:
        #     print(f"self.assemble_SinusoidalPosEmb_2d.shape: {self.assemble_SinusoidalPosEmb_2d.shape}")
        #     print(f"self.freqs_cis.shape: {self.freqs_cis.shape}")
        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)

        if self.is_adaLN:
            # 遍历所有 ConditionedEnhancementBlock 实例
            blocks_to_check = self.layers_inputreorder
            # 如果 self.layers 也是 ConditionedEnhancementBlock，需要加入 self.layers
            for block in blocks_to_check:
                if isinstance(block, ConditionedEnhancementBlock):
                    # 找到 adaLN_modulation 中的最后一个 nn.Linear
                    # 它应该是 nn.Sequential 中的第二个元素
                    final_linear = block.adaLN_modulation[1]

                    # DiT 核心初始化: 将权重和偏置置为零
                    nn.init.constant_(final_linear.weight, 0)
                    nn.init.constant_(final_linear.bias, 0)

                    # 如果启用了 pass-aware adaLN，也初始化 adaLN_modulation_pass
                    if self.use_pass_aware_adaLN and hasattr(block, 'adaLN_modulation_pass'):
                        final_linear_pass = block.adaLN_modulation_pass[1]
                        nn.init.constant_(final_linear_pass.weight, 0)
                        nn.init.constant_(final_linear_pass.bias, 0)

        if self.use_conditioned_blocks:
            # 初始化主 transformer blocks 中的 ConditionedEnhancementBlock
            for block in self.layers:
                if isinstance(block, ConditionedEnhancementBlock):
                    final_linear = block.adaLN_modulation[1]
                    nn.init.constant_(final_linear.weight, 0)
                    nn.init.constant_(final_linear.bias, 0)

    def _init_mlp_weights(self):
        """
        initialize the 2nd layer of MLP to 0
        so that at the beginning, the target-aware positional embedding contribution is zero
        """
        if hasattr(self, 'pos_emb_mlp'):
            nn.init.constant_(self.pos_emb_mlp[2].weight, 0)
            nn.init.constant_(self.pos_emb_mlp[2].bias, 0)
        else:
            raise ValueError("The model does not have pos_emb_mlp attribute")

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    @torch.no_grad()
    def precompute_knn_plan_all_passes(self, decoded_masked_coords: list[torch.Tensor]):
        all_knn_idxs = []
        all_knn_ws = []
        for pass_i, query_xy in enumerate(decoded_masked_coords):
            if pass_i == 0:
                continue
            # if dist.get_rank() == 0:
            #     logging.info(f"decoded_masked_coords[:pass_i] {decoded_masked_coords[:pass_i]} / pass_i {pass_i}")
            known_xy = torch.cat(decoded_masked_coords[:pass_i], dim=0)
            idx, w = self.precompute_knn_plan_one_passes(known_xy.to(dtype=torch.float32), query_xy.to(dtype=torch.float32), k=3, weight="idw", p=2.0, eps=1e-8, chunk_q=65536)
            all_knn_idxs.append(idx)
            all_knn_ws.append(w)
        
        return all_knn_idxs, all_knn_ws
    
    @torch.no_grad()
    def precompute_knn_plan_one_passes(self, known_xy: torch.Tensor,  # (N,2) float32 [x,y]
                                            query_xy: torch.Tensor,  # (M,2) float32 [x,y]
                                            *, k: int = 3,
                                            weight: str = "idw",
                                            p: float = 2.0, sigma: float = 3.0, eps: float = 1e-8,
                                            chunk_q: Optional[int] = None
                                        ):
        
        N = known_xy.size(0); k = min(k, N); assert k > 0 and N > 0
        if chunk_q is None:
            d = torch.cdist(query_xy, known_xy)                # (M,N)
            d_k, idx = torch.topk(d, k=k, dim=-1, largest=False)  # (M,k)
        else:
            M = query_xy.size(0); dks, idxs = [], []
            for s in range(0, M, chunk_q):
                e = min(s + chunk_q, M)
                dk, ix = torch.topk(torch.cdist(query_xy[s:e], known_xy), k=k, dim=-1, largest=False)
                dks.append(dk); idxs.append(ix)
            d_k, idx = torch.cat(dks, 0), torch.cat(idxs, 0)

        if weight in ("idw", "shepard"):
            near0 = (d_k <= eps)
            if near0.any():
                z = near0.sum(-1, keepdim=True).clamp_min(1.0)
                w = torch.where(near0, 1.0 / z, torch.zeros_like(d_k))
                mask = (near0.sum(-1) == 0)
                if mask.any():
                    di = d_k[mask]; wi = (di + eps).pow(-p); wi /= (wi.sum(-1, True) + eps); w[mask] = wi
            else:
                w = (d_k + eps).pow(-p); w /= (w.sum(-1, True) + eps)
        elif weight == "rbf":
            w = torch.exp(-(d_k ** 2) / (2.0 * (sigma ** 2) + 1e-12)); w /= (w.sum(-1, True) + eps)
        else:
            raise ValueError(f"Unknown weight '{weight}'")
        return idx, w

    @torch.no_grad()
    def apply_knn_plan(self,
        idx: torch.Tensor,    # (M,k)
        w:   torch.Tensor,    # (M,k)
        known_f: torch.Tensor # (B,N,C)
    ) -> torch.Tensor:
        # B, N, C = known_f.shape
        # M, k = idx.shape
        # ix = idx.view(1, M, k, 1).expand(B, M, k, C)     # (B,M,k,C)
        # nnf = torch.gather(known_f, dim=1, index=ix)     # (B,M,k,C)
        # return (nnf * w.view(1, M, k, 1)).sum(dim=2)     # (B,M,C)
        B, N, C = known_f.shape; M, k = idx.shape
    
        # 1. 展平索引，并扩展 Batch 维度 (B, M*k)
        idx_flat = idx.view(-1).unsqueeze(0).expand(B, M * k) # (B, M*k)
        
        # 2. 扩展 C 维度：将 (B, M*k) 索引复制 C 次，以便在 C 维度上进行 gather
        # 这样索引张量 (B, M*k, C) 就与 known_f (B, N, C) 有相同的维度数 3
        idx_b_mk_c = idx_flat.unsqueeze(-1).repeat(1, 1, C).long() # (B, M*k, C)
        
        # 3. 执行 gather (在 dim=1，即 N 维度上查找)
        # nnf_temp 形状为 (B, M*k, C)
        nnf_temp = torch.gather(known_f, dim=1, index=idx_b_mk_c) 
        
        # 4. 恢复 (B, M, k, C) 形状
        nnf = nnf_temp.view(B, M, k, C) # (B, M, k, C)
        
        # 5. 执行加权求和
        w_4d = w.view(1, M, k, 1) # (1, M, k, 1)
        return (nnf * w_4d).sum(dim=2) # (B, M, C)
    
    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        # causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        causal_mask = torch.zeros(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        grid_size = self.auto_regr_struct.training_attention_mask.shape[0]
        grid_size = int(grid_size)
        # if dist.get_rank() == 0:
        #     print(f"grid_size: {grid_size}, self.max_seq_length: {self.max_seq_length}")
        causal_mask[:grid_size, :grid_size] = self.auto_regr_struct.training_attention_mask
        causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        self.register_buffer('causal_mask', causal_mask)
        # grid_size = int(self.config.block_size ** 0.5)
        # assert grid_size * grid_size == self.block_size
        # self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)

    def forward(
        self, 
        idx: torch.Tensor, 
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        input_pos:  Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        ):
        if idx is not None and cond_idx is not None:
            return self.forward_train(idx, cond_idx, input_pos, targets, valid)
        else:
            raise ValueError("idx and cond_idx cannot be both None")

    def forward_inference(self, 
                          x: torch.Tensor,
                          cond_embeddings: torch.Tensor,
                          freqs_cis: torch.Tensor, 
                          input_pos: torch.Tensor,
                          pass_i: int = None,
                          cond_emb_adaLN: Optional[torch.Tensor] = None,):
        """ Args:
            x: [bs, query_num, dim] Input tokens
            freqs_cis: [bs, query_num, n_head, dim // n_head] Frequency embeddings
            input_pos: [query_num] Position index for each token
        """

        # TODO: add support for KV cache using input_pos 
        # if dist.get_rank() == 0:
        #     print("^"*50)
        #     print(f"input_pos inside forward_inference: {input_pos}")
        #     print(f"cond_embeddings.shape inside forward_inference: {cond_embeddings.shape}")
        #     print(f"cond_embeddings.dtype inside forward_inference: {cond_embeddings.dtype}")
        #     print(f"x.shape inside forward_inference: {x.shape}")
        #     print(f"x.dtype inside forward_inference: {x.dtype}")
        #     print(f"freqs_cis.shape inside forward_inference: {freqs_cis.shape}")
        #     print("^"*50)
        
        assert self.auto_regr_struct.training_attention_mask is not None
        # mask = self.auto_regr_struct.training_attention_mask[:x.shape[1], :x.shape[1]].to(x.device)
        # if dist.get_rank() == 0:
        #     print(f"self.assemble_SinusoidalPosEmb.device: {self.assemble_SinusoidalPosEmb.device}")
        #     print(f"self.assemble_SinusoidalPosEmb.dtype: {self.assemble_SinusoidalPosEmb.dtype}")
        #     print(f"input_pos.device: {input_pos.device}")
        #     print(f"input_pos.dtype: {input_pos.dtype}")
        #     print(f"self.causal_mask.device: {self.causal_mask.device}")
        #     print(f"self.causal_mask.dtype: {self.causal_mask.dtype}")
        #     print(f"x.dtype: {x.dtype}")
        #     print(f"freqs_cis.dtype: {freqs_cis.dtype}")
            
        mask = self.causal_mask[:x.shape[0], None, input_pos]
        # if dist.get_rank() == 0:
        #     print(f'mask.shape inside forward_inference: {mask.shape}')
        #     print(f'mask[0].shape inside forward_inference: {mask.shape[0]}')
        #     visualize_attention_mask(
        #         attention_mask=mask[0, 0],
        #         experiment_dir=self.sample_folder_dir,
        #         mask_i = pass_i
        #     )
        h = x
        if cond_emb_adaLN is None:
            cond_emb_adaLN = cond_embeddings
            
        if self.target_aware_emb:
            pos_emb = self.assemble_SinusoidalPosEmb_2d
            delta_h = self.pos_emb_mlp(pos_emb) 
            h = h + delta_h[input_pos].unsqueeze(0)

        if pass_i >= self.first_pass_token_num: # the first pass is generated token-by-token, so it does not need input reordering
            if self.pre_token_choose == 'knn' and h.shape[1] > 1:
                idx_name = self.knn_idx_names[int(pass_i - self.first_pass_token_num)]
                w_name = self.knn_w_names[int(pass_i - self.first_pass_token_num)]
                knn_idx = getattr(self, idx_name) # 这将返回 CUDA Tensor
                knn_w = getattr(self, w_name)
                token_num_curr_pass = knn_idx.shape[0]
                known_f = h  # (B,N,C)
                pred_feat = self.apply_knn_plan(knn_idx, knn_w, known_f)  # (B,M,C) in our case, M=N
                h = pred_feat
            elif self.pre_token_choose == 'transformer_choose' and h.shape[1] > 1:
                # Calculate which module to use (shared module for multiple passes)
                pass_idx = pass_i - self.first_pass_token_num  # 0-indexed pass (excluding first pass)
                module_idx = pass_idx // self.passes_per_module

                # In inference, use the actual token count from h (current pass only)
                # The module is shared, but each pass is processed separately during inference
                token_num_curr_pass = h.shape[1]

                h_curr_pass = h  # (B, curr_pass_token_num, C)
                if self.is_adaLN:
                    # Get pass embedding if enabled
                    pass_emb = None
                    if self.use_pass_aware_adaLN:
                        batch_size = h.shape[0]
                        pass_idx_tensor = torch.tensor([pass_i], device=h.device).expand(batch_size)
                        pass_emb = self.pass_embedding(pass_idx_tensor)  # (B, 1, C)
                        

                    h = self.layers_inputreorder[module_idx](h_curr_pass, cond_embeddings,
                                    self.freqs_cis[self.cls_token_num:self.cls_token_num+token_num_curr_pass].to(h.device),
                                    None,
                                    mask=torch.ones((token_num_curr_pass, token_num_curr_pass), dtype=torch.bool).to(h.device),
                                    pass_emb=pass_emb)
                else:
                    h = self.layers_inputreorder[module_idx](h_curr_pass,
                                    self.freqs_cis[self.cls_token_num:self.cls_token_num+token_num_curr_pass].to(h.device),
                                    None,
                                    mask=torch.ones((token_num_curr_pass, token_num_curr_pass), dtype=torch.bool).to(h.device))

        for layer in self.layers:
            # h = layer(h, freqs_cis, start_pos=input_pos, mask=self.causal_mask)
            if self.use_conditioned_blocks:
                # prepare token-wise conditioning
                batch_size, seq_len, hidden_dim = h.shape # seq_len is current pass token num

                # Expand class embedding to (B, L, C)
                cond_embeddings_expanded = cond_emb_adaLN[:, :self.cls_token_num, :].expand(batch_size, seq_len, self.class_pass_emb_dim)

                if self.use_pass_aware_adaLN:
                    # For inference, all tokens in current batch belong to the same pass
                    # pass_idx = pass_i - self.first_pass_token_num  # 0-indexed pass (excluding first pass)
                    pass_idx_tensor = torch.tensor([pass_i], device=h.device).expand(batch_size)
                    pass_emb = self.pass_embedding(pass_idx_tensor)  # (B, 1, C)
                    pass_emb_expanded = pass_emb.expand(batch_size, seq_len, self.class_pass_emb_dim)  # (B, L, C)
                    # self.logger.info(f"pass_i: {pass_i}, cond_embeddings_expanded.shape: {cond_embeddings_expanded.shape}, pass_emb_expanded.shape: {pass_emb_expanded.shape}")
                    assert pass_emb_expanded.shape == cond_embeddings_expanded.shape, \
                        f"pass_emb_expanded.shape {pass_emb_expanded.shape} != cond_embeddings_expanded.shape {cond_embeddings_expanded.shape}"
                if self.use_class_aware_adaLN and self.use_pass_aware_adaLN:
                    # both class-aware and pass-aware adaLN
                    cond_embeddings_expanded = cond_embeddings_expanded + pass_emb_expanded
                elif not self.use_class_aware_adaLN and self.use_pass_aware_adaLN:
                    # only pass-aware adaLN
                    cond_embeddings_expanded = pass_emb_expanded

                h = layer(h, cond_embeddings_expanded, freqs_cis, input_pos=input_pos, mask=mask[0, 0])
            else:
                h = layer(h, freqs_cis, start_pos=input_pos, mask=mask[0, 0])
        h = self.norm(h)
        logits = self.output(h).float()
        return logits

    def forward_train(
        self, 
        idx: torch.Tensor, 
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        input_pos:  Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ):
        assert targets is None
        cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]

        # prepare adaLN embedding of the class for ConditionedEnhancementBlock
        if self.use_class_aware_adaLN and self.class_pass_emb_dim != self.config.dim:
            cond_adaLN_emb = self.cls_adaLN_emb(cond_idx, train=self.training)[:,:self.cls_token_num]
        else:
            cond_adaLN_emb = None

        token_embeddings = self.tok_embeddings(idx)
        assem_input_embeddings, assem_freqs_cis = self.auto_regr_struct.assemble_input_tokens(token_embeddings, cond_embeddings, self.learnable_pos_embedding, self.freqs_cis.to(idx.device))

        h = self.tok_dropout(assem_input_embeddings)
        if self.target_aware_emb:
            self.assemble_SinusoidalPosEmb_2d = self.assemble_SinusoidalPosEmb_2d.to(dtype=h.dtype)
            pos_emb = self.assemble_SinusoidalPosEmb_2d
            delta_h = self.pos_emb_mlp(pos_emb) 
            h = h + delta_h.unsqueeze(0)
        
        if self.pre_token_choose == 'knn':
            for i, (idx_name, w_name) in enumerate(zip(self.knn_idx_names, self.knn_w_names)): 
                knn_idx = getattr(self, idx_name) # 这将返回 CUDA Tensor
                knn_w = getattr(self, w_name)
                token_num_curr_pass = knn_idx.shape[0]
                h_left = h[:, :token_num_curr_pass, :]  # (B, curr_pass_token_num, C)
                known_f = h[:, token_num_curr_pass: 2*token_num_curr_pass, :]  # (B, curr_pass_token_num, C)
                h_right = h[:, 2*token_num_curr_pass:, :]  # (B, rest_token_num, C)
                pred_feat = self.apply_knn_plan(knn_idx, knn_w, known_f)  # (B,M,C) in our case, M=N
                h_new = torch.cat((h_left, pred_feat, h_right), dim=1)
                h = h_new
        
        elif self.pre_token_choose == 'transformer_choose' and not self.is_adaLN:
            # self.logger.info(f"Using normal TransformerBlock in input reordering blocks during training.")
            h_before_reorder = h.clone()
            total_passes = len(self.ori_masked_coords) - 1  # exclude first pass
            # Iterate through all passes, but use shared modules
            for pass_idx in range(total_passes):
                # Determine which module to use (multiple passes share the same module)
                module_idx = pass_idx // self.passes_per_module
                layer = self.layers_inputreorder[module_idx]

                # Get token count for current pass
                token_num_curr_pass = self.ori_masked_coords[pass_idx + 1].shape[0]

                h_left = h[:, :token_num_curr_pass, :]  # (B, curr_pass_token_num, C)
                h_curr_pass = h[:, token_num_curr_pass: 2*token_num_curr_pass, :]  # (B, curr_pass_token_num, C)
                h_right = h[:, 2*token_num_curr_pass:, :]  # (B, rest_token_num, C)

                h_reordered = layer(h_curr_pass,
                                    self.freqs_cis[self.cls_token_num:self.cls_token_num+token_num_curr_pass].to(h.device),
                                    input_pos,
                                    mask=torch.ones((token_num_curr_pass, token_num_curr_pass), dtype=torch.bool).to(h.device))
                h_new = torch.cat((h_left, h_reordered, h_right), dim=1)
                h = h_new
        
        elif self.pre_token_choose == 'transformer_choose' and self.is_adaLN:
            # self.logger.info(f"Using ConditionedEnhancementBlock in input reordering during training.")
            h_before_reorder = h.clone()
            total_passes = len(self.ori_masked_coords) - 1  # exclude first pass
            # Iterate through all passes, but use shared modules
            for pass_idx in range(total_passes):
                # Determine which module to use (multiple passes share the same module)
                module_idx = pass_idx // self.passes_per_module
                layer = self.layers_inputreorder[module_idx]

                # Get token count for current pass
                token_num_curr_pass = self.ori_masked_coords[pass_idx + 1].shape[0]

                h_left = h[:, :token_num_curr_pass, :]  # (B, curr_pass_token_num, C)
                h_curr_pass = h[:, token_num_curr_pass: 2*token_num_curr_pass, :]  # (B, curr_pass_token_num, C)
                h_right = h[:, 2*token_num_curr_pass:, :]  # (B, rest_token_num, C)

                # Get pass embedding if enabled
                pass_emb = None
                if self.use_pass_aware_adaLN:
                    batch_size = h.shape[0]
                    pass_idx_tensor = torch.tensor([pass_idx], device=h.device).expand(batch_size)
                    pass_emb = self.pass_embedding(pass_idx_tensor)  # (B, 1, C)

                # self.logger.info(f"cond_embeddings.shape: {cond_embeddings.shape}")
                h_reordered = layer(h_curr_pass, cond_embeddings,
                                    self.freqs_cis[self.cls_token_num:self.cls_token_num+token_num_curr_pass].to(h.device),
                                    input_pos,
                                    mask=torch.ones((token_num_curr_pass, token_num_curr_pass), dtype=torch.bool).to(h.device),
                                    pass_emb=pass_emb)
                h_new = torch.cat((h_left, h_reordered, h_right), dim=1)
                h = h_new

            # compare = (h_before_reorder == h)
            # if dist.get_rank() == 0:
            #     print(f"total number of tokens in the first pass: {compare[:,:4,:].numel()}")
            #     print(f"After transformer-based input reordering, number of unchanged in the first pass: {compare[:,:4,:].sum().item()}")
            #     print(f"unchanged in the following pass: {compare[:,4:,:].sum().item()}")


        assert self.auto_regr_struct.training_attention_mask is not None

        # Process through backbone layers
        if self.use_conditioned_blocks:
            # When backbone uses pass-aware adaLN, prepare token-wise conditioning
            # Expand class embedding to (B, L, C)
            batch_size, seq_len, hidden_dim = h.shape
            if self.use_class_aware_adaLN and cond_adaLN_emb is not None:
                cond_embeddings_expanded = cond_adaLN_emb[:, :self.cls_token_num, :].expand(batch_size, seq_len, self.class_pass_emb_dim)  # (B, L, C)
            else:
                cond_embeddings_expanded = cond_embeddings[:, :self.cls_token_num, :].expand(batch_size, seq_len, self.class_pass_emb_dim)  # (B, L, C)

            # Prepare pass embedding for each token based on which pass it belongs to
            # total_passes = len(self.ori_masked_coords) - 1  # exclude first pass
            total_passes = len(self.auto_regr_struct.decoded_masked_coords)
            pass_embeddings_list = []

            if self.use_pass_aware_adaLN:
                for pass_idx in range(total_passes):
                    token_num_curr_pass = self.auto_regr_struct.decoded_masked_coords[pass_idx].shape[0]
                    # Get pass embedding for this pass
                    pass_idx_tensor = torch.tensor([pass_idx], device=h.device).expand(batch_size)
                    pass_emb = self.pass_embedding(pass_idx_tensor)  # (B, 1, C)
                    # Expand to match the number of tokens in this pass
                    pass_emb_expanded = pass_emb.expand(batch_size, token_num_curr_pass, self.class_pass_emb_dim)  # (B, token_num_curr_pass, C)
                    pass_embeddings_list.append(pass_emb_expanded)

                # Concatenate all pass embeddings: (B, total_tokens, C)
                pass_embeddings_expanded = torch.cat(pass_embeddings_list, dim=1)  # (B, L, C)
                assert pass_embeddings_expanded.shape == cond_embeddings_expanded.shape, \
                    f"pass_embeddings_expanded.shape {pass_embeddings_expanded.shape} != cond_embeddings_expanded.shape {cond_embeddings_expanded.shape}"

                
            if self.use_class_aware_adaLN and self.use_pass_aware_adaLN:
                # both class-aware and pass-aware adaLN
                # self.logger.info(f"cond_embeddings_expanded.shape: {cond_embeddings_expanded.shape}, pass_embeddings_expanded.shape: {pass_embeddings_expanded.shape}")
                cond_embeddings_expanded = cond_embeddings_expanded + pass_embeddings_expanded
            elif not self.use_class_aware_adaLN and self.use_pass_aware_adaLN:
                # only pass-aware adaLN
                cond_embeddings_expanded = pass_embeddings_expanded

            # Process through each layer with token-wise conditioning
            for layer in self.layers:
                h = layer(h, cond_embeddings_expanded, assem_freqs_cis, input_pos,
                         self.auto_regr_struct.training_attention_mask.to(h.device))
        else:
            # Original processing without token-wise pass-aware conditioning
            for layer in self.layers:
                h = layer(h, assem_freqs_cis, input_pos, self.auto_regr_struct.training_attention_mask.to(h.device))
        
        h = self.norm(h)
        logits = self.output(h).float()
        targets, valid = self.auto_regr_struct.assemble_target_tokens(image_token_idx=idx) 
        # if we are given some desired targets also calculate the loss
        loss = None
        if valid is not None:
            loss_all = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
            )
            valid_all = valid.view(-1) # valid[:, None].repeat(1, targets.shape[1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)

    @torch.no_grad()
    def generate(self,
                 cond_idx: torch.Tensor,
                 max_new_tokens: int,
                 cfg_scales: tuple[float, float] = (1.0, 1.0),
                 num_inference_steps: int = 88,
                 temperature: float = 1.0,
                 top_k: int = 0,
                 top_p: float = 1.0):
        
        self.freqs_cis = self.freqs_cis.to(cond_idx.device)
        assembled_freq_cis = self.auto_regr_struct.assemble_positional_embedding(self.freqs_cis)
        bs = cond_idx.shape[0]
        cond_lenn = self.cls_token_num
        ############################################################################################################################
        decoding_schedule = self.auto_regr_struct.decoding_schedule
        bs_ori = bs
        ############################################################################################################################
        ############################################################################################################################
        decoded_indices = torch.zeros((bs, len(self.auto_regr_struct.token_map)), dtype=torch.long, device=cond_idx.device)
                        # for check pass-wise image generation
        # decoded_indices = torch.zeros((bs_ori*len(decoding_schedule), len(self.auto_regr_struct.token_map)), dtype=torch.long, device=cond_idx.device)
        ############################################################################################################################
        if cfg_scales[-1] > 1.0:
            cond_null = torch.ones_like(cond_idx) * self.num_classes
            cond_combined = torch.cat([cond_idx, cond_null])
            bs *= 2
        else:
            cond_combined = cond_idx
        
    
        cond_combined_tokens = self.cls_embedding(cond_combined, train=False)

        # if adaLN embedding dimension is different from model hidden dimension, we need to project it to the same dimension for addition
        if self.use_class_aware_adaLN and self.class_pass_emb_dim != self.config.dim:
            cond_emb_adaLN = self.cls_adaLN_emb(cond_combined, train=False)
        else:
            cond_emb_adaLN = None
            
        x = torch.empty([bs, 0, self.config.dim], device=cond_idx.device, dtype=cond_combined_tokens.dtype)
        if self.target_aware_emb:
            self.assemble_SinusoidalPosEmb_2d = self.assemble_SinusoidalPosEmb_2d.to(dtype=cond_combined_tokens.dtype)
        input_pos = torch.empty(0, device=cond_idx.device, dtype=torch.long)
            
        # TODO: add support for KV cache (below code is from RandAR)
        # Step-4: KV Cache setup
        # max_seq_len = cond_combined_tokens.shape[1] + self.block_size
        max_seq_len = self.block_size
        with torch.device(cond_idx.device):
            self.setup_caches(max_batch_size=bs, max_seq_length=max_seq_len, dtype=self.tok_embeddings.weight.dtype)
        
        
        input_token_config = list(self.auto_regr_struct.token_map.input_tokens())
        output_token_config = {v: k for k, v in enumerate(self.auto_regr_struct.token_map.output_tokens())}
        
        ########################################################################
        # pre_indics = None     # for check pass-wise image generation
        # decoding_mask = []    # for check pass-wise image generation
        # rows_cols = []        # for check pass-wise image generation
        ########################################################################

        for decoding_step in range(len(decoding_schedule)):
            next_decoded_token_group = decoding_schedule[decoding_step]
            # next_embeddings = torch.zeros((decoded_indices.shape[0], len(next_decoded_token_group ), self.config.dim), dtype=x.dtype, device=x.device)
            next_embeddings = torch.zeros((bs, len(next_decoded_token_group ), self.config.dim), dtype=x.dtype, device=x.device)

            for idx, next_decoded_token_idx in enumerate(next_decoded_token_group):
                # next_decoded_token_idx是即将生成的token在output tokens中的位置
                input_token = input_token_config[next_decoded_token_idx]
                # input_token是即将生成的token的前序token
                input_token_type = input_token.token_type()
                
                if input_token_type == TokenType.IMAGE:
                    tmp = output_token_config[input_token] # tmp是前序token在output tokens中的位置
                    # next_embeddings[:, idx, :] = self.tok_embeddings(decoded_indices[:, tmp])
                    ########################################################################
                    img_emb = self.tok_embeddings(decoded_indices[:, tmp])   # [bs, dim]
                    # img_emb = self.tok_embeddings(decoded_indices[bs_ori*(decoding_step-1):bs_ori*(decoding_step), tmp])   # [bs, dim] # for check pass-wise image generation
                    ########################################################################
                    if cfg_scales[-1] > 1.0:
                        img_emb = torch.cat([img_emb, img_emb], dim=0)       # [2*bs, dim]
                    ############################################################################
                    # print(f"next_embeddings.shap: {next_embeddings.shape}, img_emb.shape: {img_emb.shape}, idx: {idx}, tmp: {tmp}")
                    # print(f"decoded_indices[bs_ori*(decoding_step-1):bs_ori*(decoding_step), tmp]: {decoded_indices[bs_ori*(decoding_step-1):bs_ori*(decoding_step), tmp]}")
                    ############################################################################
                    next_embeddings[:, idx, :] = img_emb
                elif input_token_type == TokenType.LEARNED:
                    next_embeddings[:, idx, :] = self.learnable_pos_embedding[None, None]
                elif input_token_type == TokenType.CONDITION:
                    next_embeddings[:, idx, :] = cond_combined_tokens[:, input_token.cond_index, :]
                else:
                    assert False, f"Invalid token type {input_token_type}"
                    
                

            # x = torch.cat([x, next_embeddings], dim=1)
            # input_pos = torch.cat([input_pos, torch.tensor(next_decoded_token_group, device=cond_idx.device)], dim=0)
            # freqs_cis = assembled_freq_cis[input_pos]
            
            # next_embeddings are the embeddings of the input tokens
            # input_pos should be the position index of the input tokens in the input sequence
            input_pos = torch.tensor(next_decoded_token_group, device=cond_idx.device)
            freqs_cis = assembled_freq_cis[input_pos]
            # if dist.get_rank() == 0:
            #     print(f"input_pos: {input_pos}")
            #     print(f'next_embeddings.shape[1]: {next_embeddings.shape[1]}, freqs_cis.shape[0]: {freqs_cis.shape[0]}')
            #     print(f'next_embeddings.shape: {next_embeddings.shape}, freqs_cis.shape: {freqs_cis.shape}')
        
            # assert x.shape[1] == freqs_cis.shape[0]
            assert next_embeddings.shape[1] == freqs_cis.shape[0]

            decoded_token_group = decoding_schedule[decoding_step]
            num_decoded_tokens = len(decoded_token_group)
            query_token_idx_cur_step = decoded_token_group[num_decoded_tokens // 2]
            # logits = self.forward_inference(x, freqs_cis, input_pos)
            if HAS_NEW_SDP:
                with sdpa_kernel(SDPBackend.MATH):
                    logits = self.forward_inference(next_embeddings, cond_combined_tokens, freqs_cis, input_pos, pass_i=decoding_step, cond_emb_adaLN = cond_emb_adaLN)
            else:
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                    logits = self.forward_inference(next_embeddings, cond_combined_tokens, freqs_cis, input_pos, pass_i=decoding_step, cond_emb_adaLN = cond_emb_adaLN)
            # if dist.get_rank() == 0:
            #     print(f"logits.shape: {logits.shape}, num_decoded_tokens: {num_decoded_tokens}")
            if cfg_scales[-1] > 1.0:
                cur_cfg_scale = cfg_scales[0] + (cfg_scales[-1] - cfg_scales[0]) * query_token_idx_cur_step / self.block_size
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                logits = uncond_logits + cur_cfg_scale * (cond_logits - uncond_logits)

            # if dist.get_rank() == 0:
            #     print(f"logits.shape: {logits.shape}, num_decoded_tokens: {num_decoded_tokens}")
            logits = logits[:, -num_decoded_tokens:] # [bs, query_num, vocab_size]
            # if dist.get_rank() == 0:
            #     print(f"logits.shape: {logits.shape}, num_decoded_tokens: {num_decoded_tokens}")
            
            ###################################################################################
            indices = torch.zeros(decoded_indices.shape[0], num_decoded_tokens, dtype=torch.long, device=x.device)
            # indices = torch.zeros(bs_ori, num_decoded_tokens, dtype=torch.long, device=x.device)  # for check pass-wise image generation
            ###################################################################################
            for i in range(num_decoded_tokens):
                indices[:, i : i + 1] = sample(logits[:, i : i + 1], temperature=temperature, top_k=top_k, top_p=top_p)[0]

            
            #################################################################################################
            # if pre_indics is not None:                                                            # for check pass-wise image generation
            #     decoded_indices[bs_ori*decoding_step:bs_ori*(decoding_step+1), :] += pre_indics   # for check pass-wise image generation
            # last_idx = 0                                                                          # for check pass-wise image generation
            #################################################################################################
            for idx, decoded_token_idx in enumerate(decoded_token_group):
                decoded_indices[:, decoded_token_idx] = indices[:, idx]
            #--------------------for check pass-wise image generation--------------------#
            '''
                decoded_indices[bs_ori*decoding_step:bs_ori*(decoding_step+1), decoded_token_idx] = indices[:, idx] + 1
                last_idx = decoded_token_idx
            # print(f"indices: {indices}")
            pre_indics = decoded_indices[bs_ori*decoding_step:bs_ori*(decoding_step+1)]
            # # fill the rest tokens with the last generated token
            mask = None
            if num_decoded_tokens > 1:
                image_mask = (self.auto_regr_struct.token_map_tensors.out_token_types == TokenType.IMAGE.value)
                assert image_mask.sum() == self.block_size
                decoded_indices = decoded_indices[:, image_mask]
                L = decoded_indices.shape[1]
                N = last_idx + 1 # 目前为止decode了多少个token

                _, back_order = self.auto_regr_struct.token_map_tensors.out_token_indices[image_mask].sort()
                for idxx, _ in enumerate(decoded_indices[bs_ori*decoding_step:bs_ori*(decoding_step+1)]):
                    decoded_indices[bs_ori*decoding_step:bs_ori*(decoding_step+1)][idxx] = decoded_indices[bs_ori*decoding_step:bs_ori*(decoding_step+1)][idxx][back_order]
                    mask = (decoded_indices[bs_ori*decoding_step:bs_ori*(decoding_step+1)][idxx] != 0)
                    valid_data = decoded_indices[bs_ori*decoding_step:bs_ori*(decoding_step+1)][idxx][mask]
                    rows = 2 ** int(math.log2(N) // 2) 
                    cols = 2 ** int(math.log2(N) - int(math.log2(N) // 2))
                    # print(f"Decoding step {decoding_step}, bs_ori: {bs_ori}, idxx: {idxx}, N: {N}, rows: {rows}, cols: {cols}, valid_data.shape: {valid_data.shape}")
                    valid_data_2d = valid_data.reshape(rows, cols).to(dtype=torch.float32)
                    valid_data_4d = valid_data_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, rows, cols)
                    upsampled_tensor_4d = F.interpolate(
                        valid_data_4d, 
                        size=(int(L**0.5), int(L**0.5)), 
                        mode='nearest')
                    upsampled_tensor_2d = upsampled_tensor_4d.squeeze(0).squeeze(0)
                    upsampled_tensor_1d = upsampled_tensor_2d.reshape(-1).to(dtype=torch.long)
                    decoded_indices[bs_ori*decoding_step:bs_ori*(decoding_step+1)][idxx] = upsampled_tensor_1d

                    tmp_inv = torch.argsort(back_order)
                    decoded_indices[bs_ori*decoding_step:bs_ori*(decoding_step+1)][idxx] = decoded_indices[bs_ori*decoding_step:bs_ori*(decoding_step+1)][idxx][tmp_inv] - 1
                
                decoding_mask.append(mask)
                rows_cols.append((rows, cols))

                # N = last_idx + 1
                # # L = decoded_indices.shape[1]
                # source_data = decoded_indices[bs_ori*decoding_step:bs_ori*(decoding_step+1), :N] # (bs_ori, N)
                # R = decoded_indices.shape[1] // N
                # print(f"decoded_indices[bs_ori*decoding_step:bs_ori*(decoding_step+1)].shape: {decoded_indices[bs_ori*decoding_step:bs_ori*(decoding_step+1)].shape}, source_data.shape: {source_data.shape}, R: {R}")
                # print(f"torch.repeat_interleave(source_data, repeats=R, dim=1).shape: {torch.repeat_interleave(source_data, repeats=R, dim=1).shape}")
                # decoded_indices[bs_ori*decoding_step:bs_ori*(decoding_step+1)] = torch.repeat_interleave(source_data, repeats=R, dim=1)
            # target_len = L - N
            # fill_indices = torch.arange(target_len) % N
            # filled_values = source_data[:, fill_indices]
            # decoded_indices[bs_ori*decoding_step:bs_ori*(decoding_step+1), N:] = filled_values
            '''
        
        image_mask = (self.auto_regr_struct.token_map_tensors.out_token_types == TokenType.IMAGE.value)
        assert image_mask.sum() == self.block_size
        decoded_indices = decoded_indices[:, image_mask]

        _, back_order = self.auto_regr_struct.token_map_tensors.out_token_indices[image_mask].sort()
        # if dist.get_rank() == 0:
        #     print('#'*50)
        #     print(f"back_order: {back_order}")
        #     print('#'*50)
        final_decoded_indices = decoded_indices[:, back_order]
        ###################################################################################
        # print(f"final_decoded_indices[0:8]: {final_decoded_indices[0:8]}")
        # print(f"final_decoded_indices[8:16]: {final_decoded_indices[8:16]}")
        # print(f"final_decoded_indices[16:24]: {final_decoded_indices[16:24]}")
        # print(f"final_decoded_indices[24:32]: {final_decoded_indices[24:32]}")
        # print(f"final_decoded_indices[32:40]: {final_decoded_indices[32:40]}")
        # print(f"decoded_indices[0:8]: {decoded_indices[0:8]}")
        # print(f"decoded_indices[8:16]: {decoded_indices[8:16]}")
        # print(f"decoded_indices[16:24]: {decoded_indices[16:24]}")
        # print(f"decoded_indices[24:32]: {decoded_indices[24:32]}")
        # print(f"decoded_indices[32:40]: {decoded_indices[32:40]}")
        # tmp_inv = torch.argsort(back_order)                                     # for check pass-wise image generation
        # return final_decoded_indices, decoding_mask, rows_cols, tmp_inv         # for check pass-wise image generation
        ###################################################################################

        return final_decoded_indices
            
#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache 


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 4)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    # if dist.get_rank() == 0:
    #     print("*"*50)
    #     print(f"freqs.shape inside precompute_freqs_cis_2d: {freqs.shape}")
    #     print(f"freqs[:, None, :].expand(-1, grid_size, -1).shape inside precompute_freqs_cis_2d: {freqs[:, None, :].expand(-1, grid_size, -1).shape}")
    #     print(f"freqs_grid.shape inside precompute_freqs_cis_2d: {freqs_grid.shape}")
    #     print(f"cache_grid.shape inside precompute_freqs_cis_2d: {cache_grid.shape}")
    #     print(f"cond_cache.shape inside precompute_freqs_cis_2d: {cond_cache.shape}")
    #     print("*"*50)
    return cond_cache 

def precompute_SinusoidalPosEmb(grid_size: int, dim: int, cls_token_num=120):
    seq_len = grid_size ** 2

    # for cls tokens, the sinusoidal pos emb is zero
    cls_pe = torch.zeros(cls_token_num, dim)

    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(
            torch.arange(0, dim, 2) * (- math.log(10000.0) / dim)
        )
    angles = position * div_term
    # shape of pe: [seq_len, dim]
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    final_pe = torch.cat([cls_pe, pe], dim=0) # shape of final_pe: [seq_len + cls_token_num, dim]

    return final_pe

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token_num=None):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token_num is not None:
        pos_embed = np.concatenate([np.zeros([cls_token_num, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    # if dist.get_rank() == 0:
    #     print("#"*50)
    #     print(f"freqs_cis.shape inside apply_rotary_emb: {freqs_cis.shape}")
    #     print(f"x.shape inside apply_rotary_emb: {x.shape}")
    #     print(f"so according to x, bs: {x.shape[0]}, seq_len: {x.shape[1]}, n_head: {x.shape[2]}, head_dim: {x.shape[3]}")
    #     print(f"and according to freqs_cis, seq_len: {freqs_cis.shape[0]}, head_dim // 2: {freqs_cis.shape[1]}")
    #     print("#"*50)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)



#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_7B(**kwargs):
    return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs)) # 6.6B

def GPT_3B(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs)) # 3.1B

def GPT_1B(**kwargs):
    return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs)) # 1.2B

### class-conditional
def GPT_XXXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs)) # 3.9B

def GPT_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) # 1.4B

def GPT_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

def GPT_XLcond(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, use_conditioned_blocks=True, **kwargs)) # 775M + adaLN

def GPT_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

def GPT_Lcond(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, use_conditioned_blocks=True, **kwargs)) # 343M + adaLN

def GPT_B(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M

def GPT_Bcond(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, use_conditioned_blocks=True, **kwargs)) # 111M + adaLN

def GPT_Bn1(**kwargs):
    return Transformer(ModelArgs(n_layer=13, n_head=12, dim=768, **kwargs)) # >111M
        

GPT_models = {
    'GPT-Bn1': GPT_Bn1, 'GPT-B': GPT_B, 'GPT-Bcond': GPT_Bcond, 'GPT-L': GPT_L, 'GPT-Lcond': GPT_Lcond, 'GPT-XL': GPT_XL, 'GPT-XLcond': GPT_XLcond, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B,
}




