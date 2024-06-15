import torch
from torch import Tensor, nn
from torch.nn import functional as F
from typing import Optional

# TODO
# a bunch of this will be copied / modified from
# https://github.com/PRBonn/TCoRe
# https://github.com/drapado/mot-detr/blob/main/models/transformer.py#L355
# https://github.com/cvg/LightGlue/blob/main/lightglue/lightglue.py#L133
# so sight appropriately

# TODO masking both attention and padding

# TODO add option to make normalize first True or False

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout, activation):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # TODO do I like this final norm?
    # wait this could just be the multihead attention add and norm,
    # not full encoder

    # # res = src + norm(add)
    # # value is unadded to pos encoding
    # # original is unadded to pos encoding
    # def forward(
    #     self,
    #     q_embed_0,
    #     q_embed_1,
    #     attn_mask: Optional[Tensor] = None,
    #     padding_mask: Optional[Tensor] = None,
    #     query_pos_0: Optional[Tensor] = None,
    #     query_pos_1: Optional[Tensor] = None,
    # ):
    #     q0 = k0 = self.with_pos_embed(q_embed_0, query_pos_0)
    #     q_embed_02 = self.self_attn(
    #         q0, k0, value=q_embed_0, attn_mask=attn_mask, key_padding_mask=padding_mask
    #     )[0]
    #     q_embed_0 = q_embed_0 + self.dropout(q_embed_02)
    #     q_embed_0 = self.norm(q_embed_0)

    #     q1 = k1 = self.with_pos_embed(q_embed_1, query_pos_1)
    #     q_embed_12 = self.self_attn(
    #         q1, k1, value=q_embed_1, attn_mask=attn_mask, key_padding_mask=padding_mask
    #     )[0]
    #     q_embed_1 = q_embed_1 + self.dropout(q_embed_12)
    #     q_embed_1 = self.norm(q_embed_1)

    #     return q_embed_0, q_embed_1

    # modifying above to make res like self and ff
    # res = norm(src) + add
    # value is unadded to pos encoding
    # original is unadded to pos encoding
    def forward(
        self,
        q_embed_0,
        q_embed_1,
        attn_mask_0: Optional[Tensor] = None,
        attn_mask_1: Optional[Tensor] = None,
        padding_mask_0: Optional[Tensor] = None,
        padding_mask_1: Optional[Tensor] = None,
        query_pos_0: Optional[Tensor] = None,
        query_pos_1: Optional[Tensor] = None,
    ):
        #q_embed_0 = self.norm(q_embed_0)
        #q_embed_1 = self.norm(q_embed_1)

        q0 = k0 = self.with_pos_embed(q_embed_0, query_pos_0)
        q_embed_02 = self.self_attn(
            q0, k0, value=q_embed_0, attn_mask=attn_mask_0, key_padding_mask=padding_mask_0
        )[0]
        q_embed_0 = self.norm(q_embed_0 + self.dropout(q_embed_02))

        q1 = k1 = self.with_pos_embed(q_embed_1, query_pos_1)
        q_embed_12 = self.self_attn(
            q1, k1, value=q_embed_1, attn_mask=attn_mask_1, key_padding_mask=padding_mask_1
        )[0]
        q_embed_1 = self.norm(q_embed_1 + self.dropout(q_embed_12))

        return q_embed_0, q_embed_1
    
class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout, activation):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # TODO not sure 
    # why norm behaviour is different 
    # for self and cross
    # like above is pos then norm
    # but here norm than pos
    # not sure why?

    # TODO norm q_embed_0 and q_embed_1 in same attention block?
    # or should value be un-normed?
    # original code did not (bb_feat)

    # res = norm(src) + add
    # value is added to pos encoding # TODO CHANGING THIS AS DETR DOES NOT
    # original is unadded to pos encoding
    def forward(
        self,
        q_embed_0,
        q_embed_1,
        attn_mask_0: Optional[Tensor] = None,
        attn_mask_1: Optional[Tensor] = None,
        padding_mask_0: Optional[Tensor] = None,
        padding_mask_1: Optional[Tensor] = None,
        query_pos_0: Optional[Tensor] = None,
        query_pos_1: Optional[Tensor] = None,
    ):
        #q_embed_0 = self.norm(q_embed_0)
        #q_embed_1 = self.norm(q_embed_1)

        q_embed_pos_0 = self.with_pos_embed(q_embed_0, query_pos_0)
        q_embed_pos_1 = self.with_pos_embed(q_embed_1, query_pos_1)

        q_embed_02 = self.multihead_attn(
            query=q_embed_pos_0,
            key=q_embed_pos_1,
            value=q_embed_1,
            attn_mask=attn_mask_1,
            key_padding_mask=padding_mask_1,
        )[0]

        q_embed_12 = self.multihead_attn(
            query=q_embed_pos_1,
            key=q_embed_pos_0,
            value=q_embed_0,
            attn_mask=attn_mask_0,
            key_padding_mask=padding_mask_0,
        )[0]

        q_embed_0 = self.norm(q_embed_0 + self.dropout(q_embed_02))
        q_embed_1 = self.norm(q_embed_1 + self.dropout(q_embed_12))

        return q_embed_0, q_embed_1
    
class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout, activation):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout0 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # res = norm(src) + add
    def forward(self, tgt):
        #tgt = self.norm(tgt)
        tgt2 = self.linear2(self.dropout0(self.activation(self.linear1(tgt))))
        tgt = self.norm(tgt + self.dropout1(tgt2))
        return tgt

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

# TODO no batch norm?
# TODO no dropout
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, tanh=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.tanh = tanh

    def forward(self, x):

        if self.tanh:
            for i, layer in enumerate(self.layers):
                x = torch.tanh(layer(x)) if i < self.num_layers - 1 else layer(x)

        else:
            for i, layer in enumerate(self.layers):
                x = F.leaky_relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        # if self.tanh:
        #     x = torch.tanh(x)
        
        return x