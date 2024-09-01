import torch
from torch import Tensor, nn
from torch.nn import functional as F
from typing import Optional
import copy

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm, **kwargs):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, 
                src_0,
                src_1,
                mask_0: Optional[Tensor] = None,
                mask_1: Optional[Tensor] = None,
                src_key_padding_mask_0: Optional[Tensor] = None,
                src_key_padding_mask_1: Optional[Tensor] = None,
                pos_0: Optional[Tensor] = None,
                pos_1: Optional[Tensor] = None,
                ):
        
        output_0, output_1 = src_0, src_1

        for layer in self.layers:
            output_0, output_1 = layer(output_0, 
                                       output_1, 
                                       src_mask_0=mask_0, 
                                       src_mask_1 = mask_1,
                                       src_key_padding_mask_0=src_key_padding_mask_0,
                                       src_key_padding_mask_1=src_key_padding_mask_1, 
                                       pos_0=pos_0, 
                                       pos_1=pos_1)

        if self.norm is not None:
            output_0 = self.norm(output_0)
            output_1 = self.norm(output_1)

        return output_0, output_1

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
                 d_model, 
                 nhead, 
                 dim_feedforward, # 2048
                 dropout, # 0.1
                 activation='relu',
                 normalize_before=False,
                 **kwargs,
                 ):
        super().__init__()

        # the multihead attention layer
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True 
        )

         # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_x0 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.dropout_x1 = nn.Dropout(dropout)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src_0,
                     src_1,
                     src_mask_0: Optional[Tensor] = None,
                     src_mask_1: Optional[Tensor] = None,
                     src_key_padding_mask_0: Optional[Tensor] = None,
                     src_key_padding_mask_1: Optional[Tensor] = None,
                     pos_0: Optional[Tensor] = None,
                     pos_1: Optional[Tensor] = None,
                     ):
        
        self_entries_0 = (src_0, src_mask_0, src_key_padding_mask_0, pos_0)
        self_entries_1 = (src_1, src_mask_1, src_key_padding_mask_1, pos_1)

        # self attention
        src_res = []
        for self_entry in [self_entries_0, self_entries_1]:
            src, src_mask, src_key_padding_mask, pos = self_entry

            q = k = self.with_pos_embed(src, pos)
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)

            # TODO do I do thse here? no because I do them later right? I am unsure
            # actually maybe because we pos embed both?
            src2 = self.linear2(self.dropout_x0(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        
            src_res.append(src)

        src_0, src_1 = src_res

        # cross attention
        # src_0 is query
        # src_1 is key and value
        cross_entries_0 = (src_0, src_1, src_mask_1, src_key_padding_mask_1, pos_0, pos_1)
        # src_1 is query
        # src_0 is key and value
        cross_entries_1 = (src_1, src_0, src_mask_0, src_key_padding_mask_0, pos_1, pos_0)

        src_res = []
        for cross_entry in [cross_entries_0, cross_entries_1]:
            src_q, src_kv, src_mask, src_key_padding_mask, pos_q, pos_kv = cross_entry

            q = self.with_pos_embed(src_q, pos_q)
            k = self.with_pos_embed(src_kv, pos_kv)

            src2 = self.cross_attention(q, k, value=src_kv, attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask)[0]
            src_q = src_q + self.dropout3(src2)
            src_q = self.norm3(src_q)
            src2 = self.linear4(self.dropout_x1(self.activation(self.linear3(src_q))))
            src_q = src_q + self.dropout3(src2)
            src_q = self.norm4(src_q)

            src_res.append(src_q)
        
        return src_res[0], src_res[1]
    
    def forward_pre(self,
                    src_0,
                    src_1,
                    src_mask_0: Optional[Tensor] = None,
                    src_mask_1: Optional[Tensor] = None,
                    src_key_padding_mask_0: Optional[Tensor] = None,
                    src_key_padding_mask_1: Optional[Tensor] = None,
                    pos_0: Optional[Tensor] = None,
                    pos_1: Optional[Tensor] = None,
                    ):
        
        entries_0 = (src_0, src_mask_0, src_key_padding_mask_0, pos_0)
        entries_1 = (src_1, src_mask_1, src_key_padding_mask_1, pos_1)

        src_res = []
        for entry in [entries_0, entries_1]:
            src, src_mask, src_key_padding_mask, pos = entry

            src2 = self.norm1(src)
            q = k = self.with_pos_embed(src2, pos)
            src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)

            # TODO maybe we should? Do we do this?
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout_x0(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)

            src_res.append(src)

        src_0, src_1 = src_res

        # cross attention
        # src_0 is query
        # src_1 is key and value
        cross_entries_0 = (src_0, src_1, src_mask_1, src_key_padding_mask_1, pos_0, pos_1)
        # src_1 is query
        # src_0 is key and value
        cross_entries_1 = (src_1, src_0, src_mask_0, src_key_padding_mask_0, pos_1, pos_0)

        src_res = []
        for cross_entry in [cross_entries_0, cross_entries_1]:
            src_q, src_kv, src_mask, src_key_padding_mask, pos_q, pos_kv = cross_entry

            src2 = self.norm3(src_q)
            src2 = self.cross_attention(query=self.with_pos_embed(src2, pos_q),
                                        key=self.with_pos_embed(src_kv, pos_kv),
                                        value=src_kv, attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask)[0]
            src_q = src_q + self.dropout3(src2)
            src2 = self.norm4(src_q)
            src2 = self.linear4(self.dropout_x1(self.activation(self.linear3(src2))))
            src_q = src_q + self.dropout4(src2)

            src_res.append(src_q)

        src_0, src_1 = src_res

        return src_0, src_1
    
    def forward(self,
                src_0,
                src_1,
                src_mask_0: Optional[Tensor] = None,
                src_mask_1: Optional[Tensor] = None,
                src_key_padding_mask_0: Optional[Tensor] = None,
                src_key_padding_mask_1: Optional[Tensor] = None,
                pos_0: Optional[Tensor] = None,
                pos_1: Optional[Tensor] = None,
                ):
        
        # even though I am sure what I have is right
        # following detr decoder, keeping this here in case
        if src_mask_0 is not None:
            raise RuntimeError('src mask not supported')
        
        if src_mask_1 is not None:
            raise RuntimeError('src mask is not supported')
        
        if self.normalize_before:
            return self.forward_pre(src_0, src_1,
                                    src_mask_0, src_mask_1,
                                    src_key_padding_mask_0, src_key_padding_mask_1,
                                    pos_0, pos_1)
        
        return self.forward_post(src_0, src_1,
                                 src_mask_0, src_mask_1,
                                 src_key_padding_mask_0, src_key_padding_mask_1,
                                 pos_0, pos_1)
    

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

# from detectron
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x