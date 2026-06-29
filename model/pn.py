import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
warnings.filterwarnings("ignore", message="Converting mask without torch.bool dtype to bool*")

import torch
import torch.nn as nn
import math

class RotaryPositionEncoding(nn.Module):
    def __init__(self, feature_dim, pe_type='Rotary1D'):
        super().__init__()

        self.feature_dim = feature_dim
        self.pe_type = pe_type

    @staticmethod
    def embed_rotary(x, cos, sin):
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    def forward(self, x_position):
        bsize, npoint = x_position.shape
        div_term = torch.exp(
            torch.arange(0, self.feature_dim, 2, dtype=torch.float, device=x_position.device)
            * (-math.log(10000.0) / (self.feature_dim)))
        div_term = div_term.view(1, 1, -1) # [1, 1, d]

        sinx = torch.sin(x_position * div_term)  # [B, N, d]
        cosx = torch.cos(x_position * div_term)

        sin_pos, cos_pos = map(
            lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
            [sinx, cosx]
        )
        position_code = torch.stack([cos_pos, sin_pos] , dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


class RotaryPositionEncoding3D(RotaryPositionEncoding):

    def __init__(self, feature_dim, pe_type='Rotary3D'):
        super().__init__(feature_dim, pe_type)

    @torch.no_grad()
    def forward(self, XYZ):
        '''
        @param XYZ: [B,N,3]
        @return:
        '''
        bsize, npoint, _ = XYZ.shape
        x_position, y_position, z_position = XYZ[..., 0:1], XYZ[..., 1:2], XYZ[..., 2:3]
        div_term = torch.exp(
            torch.arange(0, self.feature_dim // 3, 2, dtype=torch.float, device=XYZ.device)
            * (-math.log(10000.0) / (self.feature_dim // 3))
        )
        div_term = div_term.view(1, 1, -1)  # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term)  # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        sinx, cosx, siny, cosy, sinz, cosz = map(
            lambda feat: torch.stack([feat, feat], -1).view(bsize, npoint, -1),
            [sinx, cosx, siny, cosy, sinz, cosz]
        )

        position_code = torch.stack([
            torch.cat([cosx, cosy, cosz], dim=-1),  # cos_pos
            torch.cat([sinx, siny, sinz], dim=-1)  # sin_pos
        ], dim=-1)

        position_code = position_code[:,:,:self.feature_dim]

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code[...,-1], position_code[...,-2]

        

def apply_rope(x, sin, cos):
    """
    x:   (B, N, H, D)   query or key
    sin: (B, N, D)
    cos: (B, N, D)
    """
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rot = torch.stack([-x2, x1], dim=-1).reshape_as(x)
    return x * cos.unsqueeze(2) + x_rot * sin.unsqueeze(2)


class RoPETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, rope=None):
        # src: (B, N, D)
        q = k = src
        if rope is not None:
            sin, cos = rope
            # (B, N, H, D_head)
            B, N, D = src.shape
            H = self.self_attn.num_heads
            D_head = D // H
            q = src.view(B, N, H, D_head)
            k = src.view(B, N, H, D_head)

            q = apply_rope(q, sin, cos)
            k = apply_rope(k, sin, cos)

            q = q.reshape(B, N, D)
            k = k.reshape(B, N, D)

        src2, _ = self.self_attn(q, k, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class RoPETransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Norm & Dropout
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                rope_tgt=None, rope_mem=None):
        # ---- Cross-Attention ----
        q = tgt
        k = memory
        if rope_mem is not None:
            sin_k, cos_k = rope_mem
            sin_q, cos_q = rope_tgt
            B, N, D = memory.shape
            H = self.cross_attn.num_heads
            Dh = D // H
            qv = q.view(B, q.shape[1], H, Dh)
            kv = memory.view(B, N, H, Dh)
            qv = apply_rope(qv, sin_q, cos_q)  # 只对 query/key 用同样的旋转
            kv = apply_rope(kv, sin_k, cos_k)
            q = qv.reshape(B, q.shape[1], D)
            k = kv.reshape(B, N, D)

        tgt2, _ = self.cross_attn(q, k, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ---- FFN ----
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


# class SimplePointCloudTransformer(nn.Module):
#     def __init__(self, input_dim, tgt_dim, model_dim=120, num_heads=4, num_layers=2,
#                  dim_feedforward=512, output_dim=256, dropout=0.1):
#         super().__init__()
#         self.input_proj = nn.Linear(input_dim, model_dim)

#         # RoPE
#         self.rope = RotaryPositionEncoding3D(model_dim//num_heads)

#         # Encoder with RoPE
#         self.layers = nn.ModuleList([
#             RoPETransformerEncoderLayer(
#                 d_model=model_dim,
#                 nhead=num_heads,
#                 dim_feedforward=dim_feedforward,
#                 dropout=dropout
#             )
#             for _ in range(num_layers)
#         ])

#         # Decoder with RoPE
#         self.decoder_layers = nn.ModuleList([
#             RoPETransformerDecoderLayer(
#                 d_model=model_dim,
#                 nhead=num_heads,
#                 dim_feedforward=dim_feedforward,
#                 dropout=dropout
#             )
#             for _ in range(num_layers)
#         ])

#         self.tgt_proj = nn.Linear(tgt_dim, model_dim)
#         self.norm = nn.LayerNorm(model_dim)
#         self.output_proj = nn.Linear(model_dim, output_dim)

#     def forward(self, coords, feats, mask, tgt, tgt_coords=None):
#         """
#         coords: (B, N, 3) 点云坐标
#         feats:  (B, N, F) 点云特征
#         mask:   (B, N) bool, True for valid
#         tgt:    (B, T, D_tgt) decoder 输入
#         tgt_coords: (B, T, 3) decoder query 的 3D 坐标 (可选，如果没有就用 0)
#         """
#         # ---- Encoder ----
#         x = self.input_proj(torch.cat([coords, feats], dim=-1))  # (B, N, D)
#         padding_mask = torch.logical_not(mask.bool())  # (B, N)
#         x = torch.nan_to_num(x, nan=0.0)
#         coords = torch.nan_to_num(coords, nan=0.0)

#         rope_enc = self.rope(coords)

#         for layer in self.layers:
#             x = layer(x, src_key_padding_mask=padding_mask, rope=rope_enc)

#         memory = x

#         # ---- Decoder ----
#         tgt_seq = tgt.unsqueeze(1)
#         tgt_seq = self.tgt_proj(tgt_seq)  # (B, T, D)
#         tgt_coords = torch.zeros(tgt_seq.shape[0], tgt_seq.shape[1], 3, device=tgt_seq.device) if tgt_coords is None else tgt_coords
#         rope_tgt = self.rope(tgt_coords)

#         for layer in self.decoder_layers:
#             tgt_seq = layer(
#                 tgt_seq, memory,
#                 tgt_key_padding_mask=None,
#                 memory_key_padding_mask=padding_mask,
#                 rope_tgt=rope_tgt,
#                 rope_mem=rope_enc
#             )

#         dec_out = self.norm(tgt_seq)

#         return self.output_proj(dec_out).squeeze(1)



class SimplePointCloudTransformer(nn.Module):
    def __init__(self, input_dim, tgt_dim, model_dim=128, num_heads=4, num_layers=2, dim_feedforward=512, output_dim=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_embed = nn.Linear(3, model_dim)  # 可以用 sinusoids 替代

        # self.pos_embed = RotaryPositionEncoding3D(model_dim)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # allows input shape (B, S, E)
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.tgt_proj = nn.Linear(tgt_dim, model_dim)
        
        self.norm = nn.LayerNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, output_dim)
        # self.memory_proj = nn.Linear(model_dim, output_dim)

    def forward(self, coords, feats, mask, tgt, return_intermediate=False):
        """
        coords: (B, N, 3)
        feats:  (B, N, 6)
        mask:   (B, N), bool, True for valid
        """
        x = self.input_proj(torch.cat([coords, feats], dim=-1))  # (B, N, D)
        pos = self.pos_embed(coords)
        x = x + pos  # 加上位置信息
        # padding_mask = torch.logical_not(mask.to(torch.bool))
        padding_mask = torch.logical_not(mask.bool())
        x = torch.nan_to_num(x, nan=0.0)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)  # mask: False for valid, True for pad
        
        memory = x
        B, N, E = memory.shape
        tgt_seq = tgt.unsqueeze(1)                # (B, 1, E)
        tgt_seq = self.tgt_proj(tgt_seq)

        # memory_mask: True for valid; Transformer expects True for padding positions
        # So invert: padding_mask = ~memory_mask

        # Decode: output shape (B, 1, E)
        dec_out = self.decoder(
            tgt=tgt_seq,
            memory=memory,
            memory_key_padding_mask=padding_mask
        )  # (B, 1, E)

        dec_out = dec_out.squeeze(1)              # (B, E)
        dec_out = self.norm(dec_out)
        if return_intermediate:
            return self.output_proj(dec_out),self.memory_proj(memory)
        else:
            return self.output_proj(dec_out)


class PointCloudTransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim=128, num_heads=4, num_layers=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_embed = nn.Linear(3, model_dim)  # 可以用 sinusoids 替代

        # self.pos_embed = RotaryPositionEncoding3D(model_dim)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])

    def forward(self, coords, feats, mask):
        """
        coords: (B, N, 3)
        feats:  (B, N, 6)
        mask:   (B, N), bool, True for valid
        """
        x = self.input_proj(torch.cat([coords, feats], dim=-1))  # (B, N, D)
        pos = self.pos_embed(coords)
        x = x + pos  # 加上位置信息
        # padding_mask = torch.logical_not(mask.to(torch.bool))
        padding_mask = torch.logical_not(mask.bool())
        x = torch.nan_to_num(x, nan=0.0)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)  # mask: False for valid, True for pad
        
        return x


class PointCloudTransformerDecoder(nn.Module):
    def __init__(self,
                 model_dim: int,
                 tgt_dim: int = None,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 output_dim: int = None):
        """
        Transformer Decoder for point cloud features.
        Args:
            model_dim (int): Dimension of the input embeddings (must match encoder output).
            num_heads (int): Number of attention heads.
            num_layers (int): Number of decoder layers.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout probability.
            output_dim (int, optional): If set, applies a linear projection to this dimension.
        Inputs:
            tgt: Tensor of shape (batch_size, model_dim) -- target embeddings.
            memory: Tensor of shape (batch_size, N, model_dim) -- encoder output per point.
            memory_mask: Bool tensor of shape (batch_size, N), True for valid points.
        Returns:
            Tensor of shape (batch_size, model_dim) or (batch_size, output_dim)
        """
        super().__init__()
        # Create a single layer and stack into TransformerDecoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # allows input shape (B, S, E)
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, output_dim) if output_dim is not None else None
        self.tgt_proj = nn.Linear(tgt_dim, model_dim) if tgt_dim is not None else None

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                memory_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            tgt: Tensor of shape (B, model_dim)
            memory: Tensor of shape (B, N, model_dim)
            memory_mask: Bool tensor (B, N), True for valid, False for padding
        Returns:
            out: Tensor of shape (B, model_dim) or (B, output_dim)
        """
        # Prepare shapes for TransformerDecoder
        # Treat tgt as sequence length = 1
        B, N, E = memory.shape
        if tgt.ndim == 2:
            tgt = tgt.unsqueeze(1)                # (B, 1, E)
        tgt_seq = tgt
        if self.tgt_proj is not None:
            tgt_seq = self.tgt_proj(tgt_seq)

        # memory_mask: True for valid; Transformer expects True for padding positions
        # So invert: padding_mask = ~memory_mask
        padding_mask = ~memory_mask               # (B, N)

        # Decode: output shape (B, 1, E)
        dec_out = self.decoder(
            tgt=tgt_seq,
            memory=memory,
            memory_key_padding_mask=padding_mask
        )  # (B, 1, E)

        dec_out = dec_out.squeeze(1)              # (B, E)
        dec_out = self.norm(dec_out)

        if self.output_proj:
            return self.output_proj(dec_out)
        return dec_out

def padding(point_feat):
    max_len = max(len(feats) for feats in point_feat)
    dim = point_feat[0].shape[1]
    padded_feats = torch.zeros((len(point_feat), max_len, dim), dtype=torch.float32, device=point_feat[0].device)
    # padded_feats = torch.full(
    #     (len(point_feat), max_len, dim),
    #     float('nan'),
    #     dtype=torch.float32,
    #     device=point_feat[0].device
    # )
    mask = torch.zeros((len(point_feat), max_len), dtype=torch.bool, device=point_feat[0].device)
    for i, feats in enumerate(point_feat):
        padded_feats[i, :len(feats)] = feats
        mask[i, :len(feats)] = True
    return padded_feats, mask


def ptv3_pad_features(features: torch.Tensor, batch_index: torch.Tensor):
    """
    features: (N, C)
    batch_index: (N,) int, 表示每个 feature 属于哪个 batch
    
    return:
        padded: (B, max_len, C)
        mask:   (B, max_len)  bool, True 表示有效位置，False 表示 padding
    """
    B = int(batch_index.max().item()) + 1   # batch_size
    C = features.size(1)
    
    # 统计每个 batch 内有多少元素
    lengths = torch.bincount(batch_index, minlength=B)
    max_len = lengths.max().item()
    
    # 初始化 padded 和 mask
    padded = features.new_zeros((B, max_len, C))
    mask = torch.zeros((B, max_len), dtype=torch.bool, device=features.device)
    
    # 往对应位置填充
    for b in range(B):
        idx = (batch_index == b).nonzero(as_tuple=True)[0]
        l = idx.numel()
        if l > 0:
            padded[b, :l] = features[idx]
            mask[b, :l] = True
    
    return padded, mask



if __name__ == "__main__":
    import torch
    import numpy as np

    pointnet = SimplePointCloudTransformer(9, 256)
    decoder = PointCloudTransformerDecoder(model_dim=128)
    
    point_feat = []
    for i in range(32):
        length = np.random.randint(10, 23)
        feats = torch.rand((length, 9))  # 6 features per point
        point_feat.append(feats)
    point_feat, mask = padding(point_feat)
    query_feat = torch.rand(32, 256)
        
    
    features = pointnet(point_feat[:,:, 0:3], point_feat[:, :, 3:], mask, query_feat)
    # final_feat = decoder(query_feat, features, mask)
    
    print(features[0].shape) 