import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, channel: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()

        self.linear_1 = nn.Linear(channel, time_embed_dim)
        self.act = None
        if act_fn == "silu":
            self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample):
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)
        return sample


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb

class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    

def init_mem_mask_faceformer(max_seq_len):
    mask = torch.ones(max_seq_len, max_seq_len)
    # set the diagonal to 0
    mask = mask.masked_fill(torch.eye(max_seq_len) == 1, 0)
    return mask    

def init_bi_biased_mask_faceformer(n_head, max_seq_len, period):
    # any attention mask that is as more than 3 elements is not working
    def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))
                ratio = start
                return [start*ratio**i for i in range(n)]
            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)                   
            else:                                                 
                closest_power_of_2 = 2**math.floor(math.log2(n)) 
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.div(torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1), period, rounding_mode='floor')
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
        if i+1 < max_seq_len:
            alibi[i, i+1:] = bias[-(max_seq_len-(i+1)):].flip(dims=[0])

    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)

    return alibi

from typing import Optional, Tuple, Union, Callable
from torch.nn.modules.transformer import _get_clones
class TransformerDecoder_w_Adapter(nn.TransformerDecoder):
    """
    A transformer decoder with adapter layer.
    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-DecoderLayer in the decoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder_w_Adapter, self).__init__(decoder_layer, num_layers, norm)
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, 
                tgt: Tensor, 
                memory: Tensor, 
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, 
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                adapter: Optional[Tensor] = None,
        ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            adapter: the adapter for the decoder layer (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         adapter=adapter)
            
        if self.norm is not None:
            output = self.norm(output)

        return output
        

class TransformerDecoderLayer_w_Adapter(nn.TransformerDecoderLayer):
    """
    A single layer of the transformer decoder with adapter.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: if ``True``, then the input and output tensors are provided as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, then the input and output tensors are provided as (seq, batch, feature). Default: ``False``.
        device: the desired device of the encoder layer. Default: if ``None`` will use ``torch.device("cuda")`` if ``torch.cuda.is_available()`` else ``torch.device("cpu")``
        dtype: the desired dtype of the encoder layer. Default: if ``None`` will use ``torch.float32``
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, 
                 d_model: int, 
                 nhead: int, 
                 dim_feedforward: int = 2048, 
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, 
                 batch_first: bool = False, 
                 norm_first: bool = False,
                 device=None, dtype=None) -> None:

        # folow the original transformer decoder layer
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer_w_Adapter, self).__init__(
            d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, **factory_kwargs)

    def forward(self, 
                tgt: Tensor, 
                memory: Tensor, 
                tgt_mask: Optional[Tensor] = None, 
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, 
                memory_key_padding_mask: Optional[Tensor] = None,
                adapter: Optional[Tensor] = None,
        ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            adapter: the adapter for the decoder layer (optional).
        Shape:
            see the docs in Transformer class.
        """
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, adapter=adapter)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, adapter=adapter)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, adapter=adapter))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, adapter=adapter))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block with adapter
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], 
                  key_padding_mask: Optional[Tensor],
                  adapter: Optional[Tensor] = None,
        ) -> Tensor:
        """
        Args:
            x: [B, T, E] if batch_first else [T, B, E]
            attn_mask: [T, T]
            key_padding_mask: [B, T]
            adapter: [B, A, E] if batch_first else [A, B, E]
        Returns:
            [B, T, E] if batch_first else [T, B, E]
        """
        batch_first = self.self_attn.batch_first
        # concate adapter to key and value if it is not None
        if adapter is not None:
            x_adpt = self._concate_adapter(adapter, x, batch_first=batch_first)
        else:
            x_adpt = x

        # # original self-attention block
        # tmp = self.self_attn(x, x_adpt, x_adpt, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True, )[1]
        # # visualize attention, use sns
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # length = 100
        # fig, ax = plt.subplots(figsize=(15, 10))
        # sns.heatmap(tmp[0, :length, :length+2].detach().cpu().numpy())
        # # save to disk
        # plt.savefig('self_attention.png')

        
        x = self.self_attn(x, x_adpt, x_adpt, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False, )[0]
        return self.dropout1(x)

    # cross-attention block with adapter
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], 
                   key_padding_mask: Optional[Tensor],
                   adapter: Optional[Tensor] = None,
        ) -> Tensor:
        """
        Args:
            x: [B, T, E] if batch_first else [T, B, E]
            mem: [B, S, E] if batch_first else [S, B, E]
            attn_mask: [T, S]
            key_padding_mask: [B, T]
            adapter: [B, A, E] if batch_first else [A, B, E]
        Returns:
            [B, T, E] if batch_first else [T, B, E]
        """

        batch_first = self.multihead_attn.batch_first
        # concate adapter to key and value if it is not None
        if adapter is not None:
            mem_adpt = self._concate_adapter(adapter, mem, batch_first=batch_first)
        else:
            mem_adpt = x

        # tmp = self.multihead_attn(x, mem_adpt, mem_adpt, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True, )[1]
        # # visualize attention, use sns
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # length = 100
        # fig, ax = plt.subplots(figsize=(15, 10))
        # sns.heatmap(tmp[0, :length, :length+2].detach().cpu().numpy())
        # # save to disk
        # plt.savefig('cross_attention.png')

        # original cross-attention block
        x = self.multihead_attn(x, mem_adpt, mem_adpt, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False, )[0]
        return self.dropout2(x)
        
    def _concate_adapter(self, adapter: Tensor, x: Tensor, batch_first: bool = True):
        """
        concate adapter ahead of x
        Args:
            adapter: [B, A, E] if batch_first else [A, B, E]
            x: [B, T, E] if batch_first else [T, B, E]
        Returns:
            x_adapted: [B, A+T, E] if batch_first else [A+T, B, E]
        """
        if batch_first:
            x_adapted = torch.concat([adapter, x], dim=1) # [B, A, E] + [B, T, E] -> [B, A+T, E]  
        else: # batch_first
            x_adapted = torch.concat([adapter, x], dim=0) # [A, B, E] + [T, B, E] -> [A+T, B, E]
        return x_adapted

class Adpt_Bias_Denoiser(nn.Module):
    # this model is based on the trasnformer_adpt.py but with some modifications for the diffusion denoising task
    def __init__(self,
                 nfeats: int = 15069,
                 latent_dim: list = 174,
                 ff_size: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 arch: str = "trans_dec",
                 audio_encoded_dim: int = 768,
                 max_len: int = 600,
                 id_dim: int = 10,
                 return_intermediate_dec: bool = False,
                 flip_sin_to_cos: bool = True,
                 freq_shift: int = 0,
                 mem_attn_scale: float = 1.0,
                 tgt_attn_scale: float = 0.1,
                 period: int = 30,
                 no_cross: bool = False,
                 **kwargs) -> None:

        super().__init__()
        self.latent_dim = latent_dim
        self.arch = arch
        self.audio_encoded_dim = audio_encoded_dim

        # audio projecter
        self.audio_feature_map = nn.Linear(audio_encoded_dim, latent_dim)

        # motion projecter
        self.vertice_map = nn.Linear(nfeats, latent_dim)

        # periodic positional encoding
        self.PPE = PeriodicPositionalEncoding(latent_dim, period = period, max_seq_len=5000) # max_seq_len can be adjusted if thit reporst an error

        # attention bias
        assert mem_attn_scale in [-1.0, 0.0, 1.0]
        self.use_mem_attn_bias = mem_attn_scale != 0.0
        self.use_tgt_attn_bias = tgt_attn_scale != 0.0
        self.memory_bi_bias = init_mem_mask_faceformer(max_len)
        
        if tgt_attn_scale < 0.0: # means we only use the causal attention
            self.target_bi_bias = init_bi_biased_mask_faceformer(num_heads, max_len, period)
            mask = torch.triu(torch.ones(max_len, max_len), diagonal=1) == 1
            self.target_bi_bias = self.target_bi_bias.masked_fill(mask, float('-inf'))
        else:
            self.target_bi_bias = init_bi_biased_mask_faceformer(num_heads, max_len, period)



        # init decoder
        decoder_layer = TransformerDecoderLayer_w_Adapter(
            d_model=latent_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_size,
            dropout=dropout, 
            activation=activation, 
            norm_first=normalize_before,
            batch_first=True
        )

        self.transformer_decoder = TransformerDecoder_w_Adapter(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
            )

        # used for diffusion denoising
        self.time_proj = Timesteps(
            audio_encoded_dim, 
            flip_sin_to_cos=flip_sin_to_cos, # because baseline models is trained with this
            downscale_freq_shift=freq_shift, # same as above
        )
        self.time_embedding = TimestepEmbedding(
            audio_encoded_dim,
            latent_dim * num_layers
        )
        
        # motion decoder
        self.motion_decoder = nn.Linear(latent_dim, nfeats)
        nn.init.constant_(self.motion_decoder.weight, 0)
        nn.init.constant_(self.motion_decoder.bias, 0)

        # style embedding
        self.obj_vector = nn.Embedding(id_dim, latent_dim * num_layers, )

        # whether we do not use cross attention
        self.no_cross = no_cross

    def forward(self,
                vertice_input: torch.Tensor,
                hidden_state: torch.Tensor,
                timesteps: torch.Tensor,
                adapter: torch.Tensor = None, # conditions other than the time embedding
                tgt_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None,
                **kwargs):
        """
        Auto-regressive forward pass for the decoder.
        To be used during training.
        Args:
            vertice_input: [N, T, E]
            hidden_state: [N, S, E]
            adapter: [N, A, E]
            tgt_mask: [N * H, T, T]
            tgt_key_padding_mask: [N, T]
            memory_mask: [T, S]
            memory_key_padding_mask: [N, S]
        """
        
        # vertice projection
        vertice_input = self.vertice_map(vertice_input)
        vertice_input = self.PPE(vertice_input)

        # time projection
        time_emb = self.time_proj(timesteps).to(vertice_input.device)
        time_emb = self.time_embedding(time_emb).unsqueeze(1) # time_emb.shape = [N, 1, E]

        # treat the time embedding as an adapter
        if adapter is not None:
            adapter = torch.concat([adapter, time_emb], dim=1)
        else:
            adapter = time_emb

        vertice_out = vertice_input
        # split the adpater in to num_layers pieces, in order to feed them into the transformer
        adapters = adapter.split(self.latent_dim, dim=-1)

        # concat the hidden state and the vertice input
        if self.no_cross:
            hidden_len = hidden_state.shape[1]
            vertice_out = torch.cat([hidden_state, vertice_out], dim=1)
            hidden_state = torch.cat([hidden_state, hidden_state], dim=1)

        for mod,adapter in zip(self.transformer_decoder.layers, adapters):
            vertice_out = mod(
                tgt=vertice_out,
                memory=hidden_state,
                adapter=adapter,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask = tgt_key_padding_mask,
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                **kwargs
            )

        if self.no_cross: # remove the hidden state
            vertice_out = vertice_out[:, hidden_len:]

        if self.transformer_decoder.norm is not None:
            vertice_out = self.transformer_decoder.norm(vertice_out)

        self.transformer_decoder.layers[0].self_attn
        vertice_out = self.motion_decoder(vertice_out)

        return vertice_out