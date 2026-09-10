import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class TemporalAttentionRefinement(nn.Module):
    """
    TemporalAttentionRefinement
    ---------------------------
    A module that refines a sequence of frame features into a single
    temporal representation using multi-head attention.

    Parameters
    ----------
    input_dim : int
        Dimension of the input frame features. Default is 280.
    embed_dim : int
        Dimension of the embeddings used in multi-head attention.
        If it is different from input_dim, a linear projection is applied.
        Default is 256.
    num_heads : int
        Number of attention heads. Default is 8.
    dropout : float
        Dropout probability for multi-head attention. Default is 0.1.
    return_to_input_dim : bool
        If True, the output is projected back to the original input dimension.
        Otherwise, it remains at embed_dim. Default is False.
    """

    def __init__(
            self,
            input_dim: int = 280,
            embed_dim: int = 256,
            num_heads: int = 8,
            dropout: float = 0.1,
            return_to_input_dim: bool = False
    ):
        super(TemporalAttentionRefinement, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.return_to_input_dim = return_to_input_dim

        # Optional linear projection if embed_dim != input_dim
        if self.embed_dim != self.input_dim:
            self.input_projection = nn.Linear(self.input_dim, self.embed_dim)
        else:
            self.input_projection = None

        # Define a learnable query token
        # Shape: (1, 1, embed_dim) -> later expanded to (batch_size, 1, embed_dim)
        self.query_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        # Define the MultiheadAttention layer
        # batch_first=True: input shape is (batch_size, seq_length, embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )

        # If requested, project back to the original input dimension
        if self.return_to_input_dim:
            self.output_projection = nn.Linear(self.embed_dim, self.input_dim)
        else:
            self.output_projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for temporal attention refinement.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_frames, input_dim).

        Returns
        -------
        torch.Tensor
            The refined temporal feature of shape (batch_size, embed_dim) if
            return_to_input_dim=False, otherwise (batch_size, input_dim).
        """
        batch_size, num_frames, _ = x.shape

        # 1) Optional projection to embed_dim
        if self.input_projection is not None:
            x = self.input_projection(x)  # Shape: (batch_size, num_frames, embed_dim)

        # 2) Expand the learnable query token to match batch size
        # query_token shape after expansion: (batch_size, 1, embed_dim)
        query = self.query_token.expand(batch_size, -1, -1)

        # 3) Perform multi-head attention
        # We use the same 'x' for keys and values, and 'query' for queries.
        # Output shape: (batch_size, 1, embed_dim)
        refined_feature, _ = self.attention(query, x, x)

        # 4) Remove the sequence dimension (now that it's a single token)
        # Resulting shape: (batch_size, embed_dim)
        refined_feature = refined_feature.squeeze(dim=1)

        # 5) Optionally project back to the original dimension
        if self.output_projection is not None:
            refined_feature = self.output_projection(refined_feature)

        return refined_feature

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
        # return self.act(self.conv(x))
    
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class MultiHeadSpatialAttention(nn.Module):
    """
    A standard multi-head self-attention block for 2D spatial features.
    Input shape: (B, C, H, W)
      - B = batch size (or number of objects if each object is separate)
      - C = channels per spatial location (embedding dimension)
      - H, W = spatial height and width

    Output:
      - out: (B, C, H, W) the attended feature map
      - attn_map: (B, H, W) a single-channel map indicating how important each spatial location is overall.
    """

    def __init__(self, 
                 in_channels: int = 3,
                 num_heads: int = 4, 
                 d_model: int = None,
                 layer_norm: bool = False
                 ):
        """
        Args:
            in_channels: The number of channels in the input (C).
            num_heads: Number of attention heads.
            d_model: The dimension used for Q/K/V. If None, set d_model = in_channels.
                     Typically, d_model should be divisible by num_heads.
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_heads = num_heads
        if d_model is None:
            d_model = in_channels  # often we match input channels, but you can choose bigger or smaller
        self.d_model = d_model
        self.d_head = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Linear projections for Q, K, V
        # We'll map from in_channels -> d_model
        self.q_proj = nn.Linear(in_channels, d_model)
        self.k_proj = nn.Linear(in_channels, d_model)
        self.v_proj = nn.Linear(in_channels, d_model)

        # Final projection if we want to map back to in_channels
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.SiLU(),
            nn.Linear(in_channels, in_channels)
        )
        
        self.out_proj = nn.Linear(d_model, in_channels)

        if layer_norm:
            self.ln1 = nn.LayerNorm(in_channels)
            self.ln2 = nn.LayerNorm(in_channels)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W) input feature map

        Returns:
            out: (B, C, H, W) attended feature map
            attn_map: (B, H, W) overall spatial attention map
        """
        B, C, H, W = x.shape
        HW = H * W

        # 1) Flatten spatial dims: -> (B, H*W, C)
        #   We'll treat each (h, w) location as a token of dimension C
        x_reshaped = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x_reshaped = x_reshaped.view(B, HW, C)           # (B, HW, C)
        
        # x_reshaped = self.norm_in(x_reshaped)

        # 2) Compute Q, K, V in dimension d_model => (B, HW, d_model)
        Q = self.q_proj(x_reshaped)   # (B, HW, d_model)
        K = self.k_proj(x_reshaped)   # (B, HW, d_model)
        V = self.v_proj(x_reshaped)   # (B, HW, d_model)

        # 3) Reshape for multi-head => (B, HW, num_heads, d_head) -> transpose => (B, num_heads, HW, d_head)
        Q = Q.view(B, HW, self.num_heads, self.d_head).transpose(1, 2)  # (B, num_heads, HW, d_head)
        K = K.view(B, HW, self.num_heads, self.d_head).transpose(1, 2)  # (B, num_heads, HW, d_head)
        V = V.view(B, HW, self.num_heads, self.d_head).transpose(1, 2)  # (B, num_heads, HW, d_head)

        # 4) Attention => (B, num_heads, HW, HW)
        scale = self.d_head ** -0.5
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, num_heads, HW, HW)
        attn_weights = F.softmax(attn_logits, dim=-1)               # (B, num_heads, HW, HW)

        # 5) Weighted sum => (B, num_heads, HW, d_head)
        out_heads = torch.matmul(attn_weights, V)  # (B, num_heads, HW, d_head)

        # 6) Combine heads => (B, HW, d_model)
        out_combined = out_heads.transpose(1, 2).contiguous()  # (B, HW, num_heads, d_head)
        out_combined = out_combined.view(B, HW, self.d_model)  # (B, HW, d_model)

        # 7) Map back to in_channels => (B, HW, in_channels)
        out_mapped = self.out_proj(out_combined)  # (B, HW, in_channels)
    
        if self.layer_norm:
            out_mapped = self.ln1(out_mapped + x_reshaped)
        out = self.mlp(out_mapped)
        if self.layer_norm:
            out = self.ln2(out + out_mapped)
        # Reshape back to (B, C, H, W)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # ---- Build a Spatial Attention Map (B, H, W) for interpretability ----
        # attn_weights = (B, num_heads, HW, HW)
        # Each row i in [HW] attends to column j in [HW].
        # We want a single map that says "which location j is overall important?"

        # We'll do an "inbound attention" approach:
        #  - sum over i (the queries) => how many queries attend to pixel j
        #  - average over heads
        inbound_attn = attn_weights.sum(dim=2)  # (B, num_heads, HW)
        inbound_attn = inbound_attn.mean(dim=1) # (B, HW)
        attn_map = inbound_attn.view(B, H, W)   # (B, H, W)

        # Optionally normalize attn_map for better visualization
        # attn_map /= (attn_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-6)

        return out, attn_map


class AttentionFusion(nn.Module):
    """
    A 'Transformer-like' cross-attention module that fuses scene feature
    with multiple object features (variable N per batch) using a mask.

    Args:
        scene_dim (int): Dimension of the scene feature.
        obj_dim (int): Dimension of each object feature.
        ff_dim (int): Dimension of hidden layer in the feed-forward networks.
        num_heads (int): Number of attention heads.
        layer_norm (bool): Whether to apply LayerNorm after attention and MLP.

    Inputs:
        scene_state (Tensor): [B, scene_dim]
            - Scene-level hidden representation for each sample in the batch.
        obj_embeds (Tensor): [B, max_n, obj_dim]
            - Padded object embeddings for each sample. max_n is the
              maximum number of objects in the batch.
        mask (BoolTensor): [B, max_n]
            - A boolean mask indicating which positions are valid objects.
              True (or 1) means valid; False (or 0) means padded.

    Returns:
        fused_out (Tensor): [B, obj_dim]
            - The final cross-attended representation after residual + MLP.
        mean_attention (Tensor): [B, max_n]
            - The attention distribution over the max_n objects, averaged
              across heads.
    """

    def __init__(self, scene_dim=256, obj_dim=64, ff_dim=128,
                 num_heads=8, layer_norm: bool = False):
        super().__init__()

        self.scene_dim = scene_dim
        self.obj_dim = obj_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.layer_norm = layer_norm

        # Ensure obj_dim is divisible by num_heads
        assert obj_dim % num_heads == 0, "obj_dim must be divisible by num_heads"
        self.head_dim = obj_dim // num_heads

        # Linear transformations for Query, Key, and Value
        self.query_transform = nn.Linear(scene_dim, obj_dim)
        self.key_transform = nn.Linear(obj_dim, obj_dim)
        self.value_transform = nn.Linear(obj_dim, obj_dim)

        # Optional LayerNorm
        if self.layer_norm:
            self.ln1 = nn.LayerNorm(obj_dim)
            self.ln2 = nn.LayerNorm(obj_dim)

        # MLP modules after attention
        self.mlp_obj = nn.Sequential(
            nn.Linear(obj_dim, ff_dim),
            nn.SiLU(),
            nn.Linear(ff_dim, obj_dim),
        )

        self.mlp_fuse = nn.Sequential(
            nn.Linear(obj_dim, ff_dim),
            nn.SiLU(),
            nn.Linear(ff_dim, obj_dim),
        )

    def forward(self, scene_state, obj_embeds, mask=None):
        """
        Forward pass of AttentionFusion.

        Args:
            scene_state (Tensor): [B, scene_dim]
            obj_embeds (Tensor): [B, max_n, obj_dim]
            mask (BoolTensor): [B, max_n] or None
                - True where valid objects exist, False for padded positions.

        Returns:
            fused_out (Tensor): [B, obj_dim]
            mean_attention (Tensor): [B, max_n]
        """
        # scene_state: [B, scene_dim]
        # obj_embeds: [B, max_n, obj_dim]
        # mask: [B, max_n], True for real object, False for padded

        B, max_n, _ = obj_embeds.shape

        # 1) Cross-Attention
        # -------------------------------------------------
        # (1) Project scene_state -> query
        #     obj_embeds -> key, value
        query = self.query_transform(scene_state)  # [B, obj_dim]
        key = self.key_transform(obj_embeds)       # [B, max_n, obj_dim]
        value = self.value_transform(obj_embeds)   # [B, max_n, obj_dim]

        # (2) Reshape for multi-head attention
        # query: [B, 1, num_heads, head_dim]
        query = query.view(B, 1, self.num_heads, self.head_dim)
        # key/value: [B, max_n, num_heads, head_dim]
        key = key.view(B, max_n, self.num_heads, self.head_dim)
        value = value.view(B, max_n, self.num_heads, self.head_dim)

        # (3) Transpose to shape [B, num_heads, seq_len, head_dim]
        query = query.transpose(1, 2)  # [B, num_heads, 1, head_dim]
        key = key.transpose(1, 2)      # [B, num_heads, max_n, head_dim]
        value = value.transpose(1, 2)  # [B, num_heads, max_n, head_dim]

        # (4) Compute scaled dot-product attention:
        scale = math.sqrt(self.head_dim)
        # attention_logits: [B, num_heads, 1, max_n]
        attention_logits = torch.matmul(query, key.transpose(-2, -1)) / scale

        # (5) Masking: set padded positions to -inf
        if mask is not None:
            # mask: [B, max_n] -> unsqueeze(1,2): [B, 1, 1, max_n]
            # We'll fill where mask == False with -inf
            expanded_mask = ~mask.unsqueeze(1).unsqueeze(2)  # invert mask for masked_fill
            attention_logits = attention_logits.masked_fill(
                expanded_mask, float(0.)
            )

        # (6) Softmax over last dimension (the max_n objects)
        attention_weights = F.softmax(attention_logits, dim=-1)  # [B, num_heads, 1, max_n]

        # (7) Compute context = matmul of attention_weights and value
        # context: [B, num_heads, 1, head_dim]
        context = torch.matmul(attention_weights, value)

        # (8) Reshape back
        # context: [B, 1, num_heads, head_dim] -> [B, obj_dim]
        context = context.transpose(1, 2)               # [B, 1, num_heads, head_dim]
        context = context.reshape(B, self.obj_dim)      # [B, obj_dim]

        # 2) Residual + MLP
        # -------------------------------------------------
        # (1) First MLP block
        context_mlp = self.mlp_obj(context)

        # (2) Residual + optional LayerNorm
        context_residual = context + context_mlp
        if self.layer_norm:
            context_residual = self.ln1(context_residual)

        # (3) Second MLP block for final fusion
        fused_mlp = self.mlp_fuse(context_residual)

        # (4) Residual + optional LayerNorm
        fused_out = context_residual + fused_mlp
        if self.layer_norm:
            fused_out = self.ln2(fused_out)

        # 3) Average attention across heads
        # -------------------------------------------------
        # attention_weights: [B, num_heads, 1, max_n] -> mean over num_heads
        mean_attention = attention_weights.mean(dim=1)  # [B, 1, max_n]
        mean_attention = mean_attention.squeeze(1)      # [B, max_n]

        return fused_out, mean_attention