import torch
import copy
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np
import torch.nn.functional as F

torch.cuda.empty_cache()
torch.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


"""
This code is mainly inspired by: https://github.com/oriondollar/TransVAE and the original Transformer paper (Vaswani et al.). 
Also the shared attributed network archtecture is inspired by: https://zenodo.org/records/8255658
"""
# -----------------------------
# Dimension aliases (requested renames)
# -----------------------------
Seq_dim = 15        # formerly a_dim
ComSeq_dim = 15      # formerly ax_dim
Comp_dim = 15         # formerly x_dim
dropout = 0. 

latent_dim = Seq_dim + ComSeq_dim + Comp_dim
Comp_dim_vector = 23
Pro_dim_vector = 3 




def clones(module, N):
    """
    Produce N deep-copied identical layers.
    Adapted from: http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



def make_subsequent_mask(size: int) -> torch.Tensor:
    """
    Create an upper-triangular causal mask to hide future tokens.
    Adapted from: http://nlp.seas.harvard.edu/2018/04/03/attention.html

    Returns:
        mask (torch.BoolTensor): shape (1, size, size), True = allowed, False = masked
    """
    attn_shape = (1, size, size)
    mask_np = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask_np) == 0


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    Compute scaled dot-product attention (Vaswani et al.).
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask (0 means disallowed)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn



class ModuleListWrapper(nn.Module):
    """
    Wrap a Python list of modules into a single nn.Module with list-like access.
    (Same behavior as your ListModule; name changed only.)
    """
    def __init__(self, *modules_in):
        super().__init__()
        module_index = 0
        for submodule in modules_in:
            self.add_module(str(module_index), submodule)
            module_index += 1

    def __getitem__(self, index):
        if index < 0 or index >= len(self._modules):
            raise IndexError(f'index {index} is out of range')
        module_iter = iter(self._modules.values())
        for _ in range(index):
            next(module_iter)
        return next(module_iter)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)
    
    


def make_standard_tgt_mask(target, pad_token):
    """
    Create the standard decoder mask:
    1) padding mask
    2) subsequent (causal) mask

    Args:
        target: target token ids, shape (B, T)
        pad_token: padding token id

    Returns:
        target_mask: sequential target mask
    """
    target_mask = (target != pad_token).unsqueeze(-2)
    target_mask = target_mask & Variable(
        make_subsequent_mask(target.size(-1)).type_as(target_mask.data)
    )
    return target_mask




class Generator(nn.Module):
    "Generates token predictions after final decoder layer"
    "Adapted from: https://github.com/oriondollar/TransVAE"
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab-1)

    def forward(self, x):
        return self.proj(x)
    


class Transformer_Encoder(nn.Module):
    "Base transformer encoder architecture"
    
    def __init__(self, layer, N, d_latent, bypass_bottleneck, eps_scale):
        super().__init__()

        self.encoder_blocks = clones(layer, N)
        self.conv_bottleneck = ConvBottleneck(layer.size)

        self.latent_mean_head = nn.Linear(576, d_latent)
        self.latent_var_head = nn.Linear(576, d_latent)

        self.norm_layer = LayerNorm(layer.size)

        self.length_fc1 = nn.Linear(d_latent, d_latent * 2)
        self.length_fc2 = nn.Linear(d_latent * 2, d_latent)

        self.bypass_bottleneck = bypass_bottleneck
        self.eps_scale = eps_scale

    def predict_mask_length(self, latent_memory):
        predicted_length = self.length_fc1(latent_memory)
        predicted_length = self.length_fc2(predicted_length)
        predicted_length = F.softmax(predicted_length, dim=-1)
        predicted_length = torch.topk(predicted_length, 1)[1]
        return predicted_length

    def reparameterize(self, latent_mean, latent_logvar, eps_scale=1):
        latent_std = torch.exp(0.5 * latent_logvar)
        noise = torch.randn_like(latent_std) * eps_scale
        return latent_mean + noise * latent_std

    def forward(self, input_tokens, attention_mask):

        attention_history = []

        for block_index, encoder_block in enumerate(self.encoder_blocks):
            input_tokens, block_weights = encoder_block(input_tokens, attention_mask, return_attn=True)
            attention_history.append(block_weights.detach().cpu())

        encoded_state = self.norm_layer(input_tokens)

        if self.bypass_bottleneck:
            latent_mean = Variable(torch.tensor([0.0]))
            latent_logvar = Variable(torch.tensor([0.0]))
        else:
            encoded_state = encoded_state.permute(0, 2, 1)
            encoded_state = self.conv_bottleneck(encoded_state)
            encoded_state = encoded_state.contiguous().view(encoded_state.size(0), -1)

            latent_mean = self.latent_mean_head(encoded_state)
            latent_logvar = self.latent_var_head(encoded_state)

            encoded_state = self.reparameterize(latent_mean, latent_logvar, self.eps_scale)

            predicted_length = self.length_fc1(latent_mean)
            predicted_length = self.length_fc2(predicted_length)

        return encoded_state, latent_mean, latent_logvar, predicted_length


    
class EncoderLayer(nn.Module):
    """
    Transformer encoder layer: self-attn + feedforward with residual connections.
    Adapted from: https://github.com/oriondollar/TransVAE
    """
    def __init__(self, size, src_len, self_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.src_len = src_len
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(self.size, dropout), 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, return_attn: bool = False):
        if return_attn:
            attn = self.self_attn(x, x, x, mask, return_attn=True)
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            return self.sublayer[1](x, self.feed_forward), attn
        else:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            return self.sublayer[1](x, self.feed_forward)

class Transformer_Decoder(nn.Module):
    "Base transformer decoder architecture"

    def __init__(self, encoder_layers, decoder_layers, N, d_latent, bypass_bottleneck):
        super().__init__()

        self.post_encoder_block = clones(encoder_layers, 1)
        self.decoder_blocks = clones(decoder_layers, N)

        self.norm_layer = LayerNorm(decoder_layers.size)
        self.bypass_bottleneck = bypass_bottleneck

        self.model_dim = decoder_layers.size
        self.target_length = decoder_layers.tgt_len

        self.latent_projection = nn.Linear(d_latent, 576)
        self.deconv_expansion = DeconvBottleneck(decoder_layers.size)

    def forward(self, decoder_input, latent_memory, source_mask, target_mask):

        if not self.bypass_bottleneck:
            latent_memory = F.relu(self.latent_projection(latent_memory))
            latent_memory = latent_memory.view(-1, 64, 9)
            latent_memory = self.deconv_expansion(latent_memory)
            latent_memory = latent_memory.permute(0, 2, 1)

        for encoder_refinement in self.post_encoder_block:
            latent_memory, deconv_attention = encoder_refinement(
                latent_memory, source_mask, return_attn=True
            )

        latent_memory = self.norm_layer(latent_memory)

        cross_attention_history = []

        for block_index, decoder_block in enumerate(self.decoder_blocks):
            decoder_input, cross_attention_weights = decoder_block(
                decoder_input,
                latent_memory,
                latent_memory,
                source_mask,
                target_mask,
                return_attn=True
            )
            cross_attention_history.append(cross_attention_weights.detach().cpu())

        return self.norm_layer(decoder_input)


class DecoderLayer(nn.Module):
    """
    Transformer decoder layer:
      1) self-attn
      2) src-attn
      3) feedforward
    
    Adapted from: https://github.com/oriondollar/TransVAE
    """
    def __init__(self, size, tgt_len, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.tgt_len = tgt_len
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(self.size, dropout), 3)

    def forward(
        self,
        x: torch.Tensor,
        memory_key: torch.Tensor,
        memory_val: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        return_attn: bool = False
    ):
        m_key = memory_key
        m_val = memory_val

        if return_attn:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            src_attn = self.src_attn(x, m_key, m_val, src_mask, return_attn=True)
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m_key, m_val, src_mask))
            return self.sublayer[2](x, self.feed_forward), src_attn
        else:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m_key, m_val, src_mask))
            return self.sublayer[2](x, self.feed_forward)



############## Attention and FeedForward ################


class MultiHeadedAttention(nn.Module):
    """
    Multi-head attention module (Vaswani et al.).
    """
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, return_attn=False):
        """
        If return_attn=True, returns attention weights (same as your code).
        Otherwise returns projected output.
        """
        if mask is not None:
            mask = mask.unsqueeze(1)  # apply same mask to all heads

        nbatches = query.size(0)

        # Linear projections -> (B, h, T, d_k)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # Attention
        x, self.attn = scaled_dot_product_attention(query, key, value, mask=mask, dropout=self.dropout)

        # Concat heads
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        if return_attn:
            return self.attn
        else:
            return self.linears[-1](x)
        


class PositionwiseFeedForward(nn.Module):
    """ Position-wise feedforward block. """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.expand_layer = nn.Linear(d_model, d_ff)
        self.project_layer = nn.Linear(d_ff, d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_features):
        hidden_state = self.expand_layer(input_features)
        activated_state = F.relu(hidden_state)
        dropped_state = self.dropout_layer(activated_state)
        return self.project_layer(dropped_state)



############## BOTTLENECKS #################


class ConvBottleneck(nn.Module):
    """
    Convolutional bottleneck: reduces memory matrix to smaller representation.
    """

    def __init__(self, size):
        super().__init__()

        conv_blocks = []
        current_channels = size
        is_first_layer = True

        for layer_index in range(3):

            next_channels = int((current_channels - 64) // 2 + 64)

            if is_first_layer:
                conv_kernel = 9
                is_first_layer = False
            else:
                conv_kernel = 8

            if layer_index == 2:
                next_channels = 64

            conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(current_channels, next_channels, conv_kernel),
                    nn.MaxPool1d(2)
                )
            )

            current_channels = next_channels

        self.conv_layers = ModuleListWrapper(*conv_blocks)

    def forward(self, input_tensor):

        for conv_block in self.conv_layers:
            input_tensor = F.relu(conv_block(input_tensor))

        return input_tensor



class DeconvBottleneck(nn.Module):
    """
    Deconvolutional bottleneck: expands latent vector back into memory matrix.
    """

    def __init__(self, size):
        super().__init__()

        deconv_blocks = []
        current_channels = 64

        for layer_index in range(3):

            next_channels = (size - current_channels) // 4 + current_channels
            stride_value = 4 - layer_index
            deconv_kernel = 11

            if layer_index == 2:
                next_channels = size
                stride_value = 1

            deconv_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        current_channels,
                        next_channels,
                        deconv_kernel,
                        stride=stride_value,
                        padding=2
                    )
                )
            )

            current_channels = next_channels

        self.deconv_layers = ModuleListWrapper(*deconv_blocks)

    def forward(self, input_tensor):

        for deconv_block in self.deconv_layers:
            input_tensor = F.relu(deconv_block(input_tensor))

        return input_tensor


############## Embedding Layers ###################

    

class Embeddings(nn.Module):
    """ Token embedding layer scaled by sqrt(d_model). """

    def __init__(self, d_model, vocab):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab, d_model)
        self.model_dim = d_model

    def forward(self, token_ids):
        embedded_tokens = self.embedding_table(token_ids)
        return embedded_tokens * math.sqrt(self.model_dim)


    
class PositionalEncoding(nn.Module):
    """ Sinusoidal positional encoding. """

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()

        self.dropout_layer = nn.Dropout(p=dropout)

        positional_table = torch.zeros(max_len, d_model)
        position_index = torch.arange(0, max_len).unsqueeze(1)

        frequency_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        positional_table[:, 0::2] = torch.sin(position_index * frequency_term)
        positional_table[:, 1::2] = torch.cos(position_index * frequency_term)

        positional_table = positional_table.unsqueeze(0)

        self.register_buffer('pe', positional_table)

    def forward(self, input_tensor):
        input_tensor = input_tensor + Variable(
            self.pe[:, :input_tensor.size(1)],
            requires_grad=False
        )
        return self.dropout_layer(input_tensor)



############## Utility Layers ####################

class TorchLayerNorm(nn.Module):
    """ BatchNorm1d wrapper (kept same behavior). """

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(features)

    def forward(self, input_tensor):
        return self.batch_norm(input_tensor)



class LayerNorm(nn.Module):
    "Construct a layernorm module (manual)"
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    

class LayerNorm(nn.Module):
    """ Manual LayerNorm used by the Transformer blocks. """

    def __init__(self, features, eps=1e-6):
        super().__init__()

        self.scale_param = nn.Parameter(torch.ones(features))
        self.shift_param = nn.Parameter(torch.zeros(features))
        self.epsilon = eps

    def forward(self, input_tensor):
        mean_value = input_tensor.mean(-1, keepdim=True)
        std_value = input_tensor.std(-1, keepdim=True)

        normalized = (input_tensor - mean_value) / (std_value + self.epsilon)
        return self.scale_param * normalized + self.shift_param


class SublayerConnection(nn.Module):
    """
    Residual connection with pre-norm:
    y = x + dropout(sublayer(LN(x)))
    """

    def __init__(self, size, dropout):
        super().__init__()

        self.norm_layer = LayerNorm(size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_tensor, sublayer):
        normalized_input = self.norm_layer(input_tensor)
        sublayer_output = sublayer(normalized_input)
        dropped_output = self.dropout_layer(sublayer_output)
        return input_tensor + dropped_output


class vaeModel(nn.Module):
    def __init__(self):
        super(vaeModel, self).__init__()

        self.N = 2
        self.d_model = 128
        self.d_ff = 128
        self.d_latent = Seq_dim + ComSeq_dim
        self.h = 2
        self.dropout = 0
        self.src_len = 126
        self.tgt_len = 125
        self.EPS_SCALE = 1
        self.vocab_size = 64
        self.pad_idx = 62
        self.bypass_bottleneck = False

        self.c = copy.deepcopy
        self.attn = MultiHeadedAttention(self.h, self.d_model)
        self.ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        self.position = PositionalEncoding(self.d_model, self.dropout)

        self.Transformer_Encoder = Transformer_Encoder(
            EncoderLayer(self.d_model, self.src_len, self.c(self.attn), self.c(self.ff), self.dropout),
            self.N, self.d_latent, self.bypass_bottleneck, self.EPS_SCALE
        )

        self.Transformer_Decoder = Transformer_Decoder(
            EncoderLayer(self.d_model, self.src_len, self.c(self.attn), self.c(self.ff), self.dropout),
            DecoderLayer(self.d_model, self.tgt_len, self.c(self.attn), self.c(self.attn), self.c(self.ff), self.dropout),
            self.N, self.d_latent, self.bypass_bottleneck
        )

        self.src_embed = nn.Sequential(Embeddings(self.d_model, self.vocab_size), self.c(self.position))
        self.tgt_embed = nn.Sequential(Embeddings(self.d_model, self.vocab_size), self.c(self.position))
        self.generator = Generator(self.d_model, self.vocab_size)

        # -----------------------------
        # Composition encoder (formerly x_encoder)
        # -----------------------------
        Comp_encoder_hidden_dim = [Comp_dim_vector, 128, 128, 128, ComSeq_dim + Comp_dim]

        self.Comp_encoder_fc1 = nn.Linear(Comp_encoder_hidden_dim[0], Comp_encoder_hidden_dim[1])
        self.Comp_encoder_fc2 = nn.Linear(Comp_encoder_hidden_dim[1], Comp_encoder_hidden_dim[2])
        self.Comp_encoder_fc3 = nn.Linear(Comp_encoder_hidden_dim[2], Comp_encoder_hidden_dim[3])
        self.Comp_encoder_fc4 = nn.Linear(Comp_encoder_hidden_dim[3], Comp_encoder_hidden_dim[4])

        self.Comp_mu_head = nn.Linear(ComSeq_dim + Comp_dim, ComSeq_dim + Comp_dim)
        self.Comp_logvar_head = nn.Linear(ComSeq_dim + Comp_dim, ComSeq_dim + Comp_dim)

        # -----------------------------
        # Composition decoder (formerly x_decoder)
        # -----------------------------
        Comp_decoder_hidden_dim = [ComSeq_dim + Comp_dim, 128, 128, 128, Comp_dim_vector]

        self.Comp_decoder_fc1 = nn.Linear(Comp_decoder_hidden_dim[0], Comp_decoder_hidden_dim[1])
        self.Comp_decoder_fc2 = nn.Linear(Comp_decoder_hidden_dim[1], Comp_decoder_hidden_dim[2])
        self.Comp_decoder_fc3 = nn.Linear(Comp_decoder_hidden_dim[2], Comp_decoder_hidden_dim[3])
        self.Comp_decoder_fc4 = nn.Linear(Comp_decoder_hidden_dim[3], Comp_decoder_hidden_dim[4])

        self.activation = nn.ReLU()

    def reparameterize(self, mu, logvar):
        """
        Reparameterization used by this model’s combined latent.
        NOTE: Kept exact original behavior (sigma = exp(logvar), not exp(0.5*logvar)).
        """
        sigma = torch.exp(logvar)
        eps = torch.FloatTensor(logvar.size()[0], 1).normal_(0, 1).to(device)
        eps = eps.expand(sigma.size())
        return mu + sigma * eps

    def encoder(self, src, src_mask, Comp):
        """
        Encode:
          - sequence tokens (src) -> Seq_mu, Seq_log_var
          - composition vector (Comp) -> Comp_mu, Comp_log_var
        Then fuse into final (mu, logvar) and sample z.
        """
        # Encode input sequence (formerly adj_*)
        Seq_z, Seq_mu, Seq_log_var, Seq_pred_len = self.Transformer_Encoder(self.src_embed(src), src_mask)

        # Encode composition features (formerly x pipeline)
        Comp = self.activation(self.Comp_encoder_fc1(Comp))
        Comp = self.activation(self.Comp_encoder_fc2(Comp))
        Comp = self.activation(self.Comp_encoder_fc3(Comp))
        Comp = self.Comp_encoder_fc4(Comp)

        Comp_mu = self.Comp_mu_head(Comp)
        Comp_log_var = self.Comp_logvar_head(Comp)

        # Concatenate latent stats from Seq and Comp branches (kept original structure)
        mu_tot = torch.cat((Seq_mu, Comp_mu), dim=-1)
        log_var_tot = torch.cat((Seq_log_var, Comp_log_var), dim=-1)

        mu = torch.zeros([len(mu_tot), Seq_dim + ComSeq_dim + Comp_dim]).to(device)
        log_var = torch.zeros([len(mu_tot), Seq_dim + ComSeq_dim + Comp_dim]).to(device)

        # Same slicing/mixing logic as original code, now using renamed dims
        mu[:, :Seq_dim] = mu_tot[:, :Seq_dim]
        mu[:, Seq_dim:(Seq_dim + ComSeq_dim)] = 0.5 * (mu_tot[:, Seq_dim:Seq_dim + ComSeq_dim] + mu_tot[:, Seq_dim + ComSeq_dim:Seq_dim + 2 * ComSeq_dim])
        mu[:, Seq_dim + ComSeq_dim:] = mu_tot[:, Seq_dim + 2 * ComSeq_dim:]

        log_var[:, :Seq_dim] = log_var_tot[:, :Seq_dim]
        log_var[:, Seq_dim:(Seq_dim + ComSeq_dim)] = 0.5 * (log_var_tot[:, Seq_dim:Seq_dim + ComSeq_dim] + log_var_tot[:, Seq_dim + ComSeq_dim:Seq_dim + 2 * ComSeq_dim])
        log_var[:, Seq_dim + ComSeq_dim:] = log_var_tot[:, Seq_dim + 2 * ComSeq_dim:]

        z = self.reparameterize(mu, log_var)

        return z, mu, log_var, Seq_pred_len

    def decoder(self, z, src_mask, tgt, tgt_mask):
        """
        Decode:
          - Sequence part from z -> token logits
          - Composition part from z -> composition distribution (softmax)
        """
        Seq_de_input = z[:, :Seq_dim + ComSeq_dim]
        Comp_de_input = z[:, Seq_dim:]

        # Decode sequence (formerly adj)
        Seq_decoded = self.Transformer_Decoder(self.tgt_embed(tgt), Seq_de_input, src_mask, tgt_mask)
        Seq_out = self.generator(Seq_decoded)

        # Decode composition (formerly x decoder)
        Comp = self.activation(self.Comp_decoder_fc1(Comp_de_input))
        Comp = self.activation(self.Comp_decoder_fc2(Comp))
        Comp = self.activation(self.Comp_decoder_fc3(Comp))
        Comp = self.Comp_decoder_fc4(Comp)

        Comp_out = torch.softmax(Comp, dim=-1)
        return Seq_out, Comp_out



class Pmodel(nn.Module):
    def __init__(self):
        super(Pmodel, self).__init__()

        self.dropout_p = 0.0

        # Same hidden dims as original cModel
        self.P_hidden_dim = [latent_dim, 128, 128, 128]

        # Explicit Sequential (no loop), same logic as original
        self.net = nn.Sequential(

            # e_fc1 : latent_dim -> latent_dim
            nn.Linear(latent_dim, self.P_hidden_dim[0], bias=False),
            nn.ReLU(),

            # e_fc2 : latent_dim -> 128
            nn.Linear(self.P_hidden_dim[0], self.P_hidden_dim[1], bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),

            # e_fc3 : 128 -> 128
            nn.Linear(self.P_hidden_dim[1], self.P_hidden_dim[2], bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),

            # e_fc4 : 128 -> 128
            nn.Linear(self.P_hidden_dim[2], self.P_hidden_dim[3], bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),

            # de_out : 128 -> Pro_dim_vector
            nn.Linear(self.P_hidden_dim[3], Pro_dim_vector, bias=False),
            nn.Sigmoid()
        )

    def forward(self, latent_vector):
        return self.net(latent_vector)




def refine_composition(composition_tensor, threshold=1e-5):
    """
    Refines a composition tensor by:
    - Zeroing out small values below the threshold.
    - Adding the removed mass to the component with the highest concentration.

    Args:
        composition_tensor (torch.Tensor): shape (..., num_components)
        threshold (float): Values below this will be removed.

    Returns:
        torch.Tensor: Refined composition tensor, same shape as input.
    """
    refined = composition_tensor.clone()

    small_mask = refined < threshold
    small_sum = refined[small_mask].sum(dim=-1, keepdim=True)

    refined[small_mask] = 0.0

    max_indices = torch.argmax(refined, dim=-1, keepdim=True)

    one_hot = torch.zeros_like(refined)
    one_hot.scatter_(-1, max_indices, 1.0)

    refined += small_sum * one_hot

    return refined



def greedy_decode_reconstruction(model, latent_memory, source_mask, use_gpu=False):

    start_token = 0
    max_steps = 125

    current_tokens = torch.ones(latent_memory.shape[0], 1).fill_(start_token).long()
    full_sequence = torch.ones(latent_memory.shape[0], max_steps + 1).fill_(start_token).long()

    if use_gpu:
        source_mask = source_mask.cuda()
        current_tokens = current_tokens.cuda()
        full_sequence = full_sequence.cuda()

    model.eval()

    for step in range(max_steps):

        causal_mask = Variable(make_subsequent_mask(current_tokens.size(1)).long())

        if use_gpu:
            causal_mask = causal_mask.cuda()

        logits, decoded_states = model.decoder(
            latent_memory,
            source_mask,
            Variable(current_tokens),
            causal_mask
        )

        token_probs = F.softmax(logits[:, step, :], dim=-1)

        _, predicted_token = torch.max(token_probs, dim=1)
        predicted_token += 1

        full_sequence[:, step + 1] = predicted_token

        predicted_token = predicted_token.unsqueeze(1)
        current_tokens = torch.cat([current_tokens, predicted_token], dim=1)

    final_tokens = full_sequence[:, 1:]

    return final_tokens, decoded_states



def greedy_decode_inference(model, latent_memory, source_mask=None, use_gpu=False):

    start_token = 0
    max_steps = 125
    source_length = 126

    current_tokens = torch.ones(latent_memory.shape[0], 1).fill_(start_token).long()
    full_sequence = torch.ones(latent_memory.shape[0], max_steps + 1).fill_(start_token).long()

    if source_mask is None:
        predicted_lengths = model.Transformer_Encoder.predict_mask_length(
            latent_memory[:, :Seq_dim + ComSeq_dim]
        )

        source_mask = torch.zeros((latent_memory.shape[0], 1, source_length + 1))

        for idx in range(predicted_lengths.shape[0]):
            valid_length = predicted_lengths[idx].item()
            source_mask[idx, :, :valid_length] = torch.ones((1, 1, valid_length))

    if use_gpu:
        source_mask = source_mask.cuda()
        current_tokens = current_tokens.cuda()
        full_sequence = full_sequence.cuda()

    model.eval()

    for step in range(max_steps):

        causal_mask = Variable(
            make_subsequent_mask(current_tokens.size(1)).long()
        )

        if use_gpu:
            causal_mask = causal_mask.cuda()

        logits, decoded_states = model.decoder(
            latent_memory,
            source_mask,
            Variable(current_tokens),
            causal_mask
        )

        token_probs = F.softmax(logits[:, step, :], dim=-1)

        _, predicted_token = torch.max(token_probs, dim=1)
        predicted_token += 1

        full_sequence[:, step + 1] = predicted_token

        predicted_token = predicted_token.unsqueeze(1)
        current_tokens = torch.cat([current_tokens, predicted_token], dim=1)

    final_tokens = full_sequence[:, 1:]

    return final_tokens, decoded_states


