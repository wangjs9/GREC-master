import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from models.common_layer import EncoderLayer, DecoderLayer, DecoderLayerContextV, ComplexEmoAttentionLayer, LayerNorm , \
    _gen_bias_mask ,_gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, \
    _get_attn_subsequent_mask,  get_input_from_batch, top_k_top_p_filtering, get_output_from_batch
import config
import pprint
pp = pprint.PrettyPrinter(indent=1)
import os
import random
from sklearn.metrics import accuracy_score

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, universal=False):
        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length) if use_mask else None,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)

        if self.universal:
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

        if config.act:
            self.act_fn = ACT_basic(hidden_size)
            self.remainders = None
            self.n_updates = None

    def forward(self, inputs, mask):
        # Add input dropout
        inputs = self.input_dropout(inputs) # (batch_size, seq_len, embed_dim)
        # Project to hidden size
        inputs = self.embedding_proj(inputs) # (batch_size, seq_len, hidden_size)

        if self.universal:
            if config.act:
                inputs, (self.remainders, self.n_updates) = self.act_fn(inputs, inputs, self.enc, self.timing_signal,
                                                                   self.position_signal, self.num_layers)
                y = self.layer_norm(inputs)
            else:
                for l in range(self.num_layers):
                    inputs += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    inputs += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    inputs = self.enc(inputs, mask=mask)
                    # inputs = torch.mul(self.enc(inputs, mask=mask), cazprob + 1)
                y = self.layer_norm(inputs)
        else:
            # Add timing signal
            inputs += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                inputs = self.enc[i](inputs, mask)

            y = self.layer_norm(inputs)

        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length),  # mandatory
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        if self.universal:
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)], 0)
        # Add input dropout
        inputs = self.input_dropout(inputs)
        inputs = self.embedding_proj(inputs)

        if self.universal:
            if config.act:
                inputs, attn_dist, (self.remainders, self.n_updates) = self.act_fn(inputs, inputs, self.dec, self.timing_signal,
                                                                              self.position_signal, self.num_layers,
                                                                              encoder_output, decoding=True)
                y = self.layer_norm(inputs)

            else:
                inputs += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    inputs += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    inputs, _, attn_dist, _ = self.dec((inputs, encoder_output, [], (mask_src, dec_mask)))
                y = self.layer_norm(inputs)
        else:
            # Add timing signal
            inputs += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, attn_dist, _ = self.dec((inputs, encoder_output, [], (mask_src, dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)
        return y, attn_dist


class DecoderContextV(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(DecoderContextV, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length),  # mandatory
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        if self.universal:
            self.dec = DecoderLayerContextV(*params)
        else:
            self.dec = nn.Sequential(*[DecoderLayerContextV(*params) for l in range(num_layers)])

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, v, mask):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)], 0)
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.dec, self.timing_signal,
                                                                              self.position_signal, self.num_layers,
                                                                              encoder_output, decoding=True)
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x, _, _, attn_dist, _ = self.dec((x, encoder_output, v, [], (mask_src, dec_mask)))
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, _, attn_dist, _ = self.dec((x, encoder_output, v, [], (mask_src, dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)
        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(self, x):
        logit = self.proj(x)
        return F.log_softmax(logit,dim=-1)


class VAESampling(nn.Module):
    def __init__(self, hidden_dim, posterior_hidden_dim, out_dim=32):
        super().__init__()
        self.positive_emotions = [11, 16, 6, 8, 3, 1, 28, 13, 31, 17, 24, 0, 27]
        self.negative_emotions = [9, 4, 2, 14, 30, 29, 25, 15, 10, 23, 19, 18, 21, 7, 20, 5, 26, 12,
                                  22]  # anticipation is negative
        self.positive_emotions_t = torch.LongTensor(self.positive_emotions).to(config.device)
        self.negative_emotions_t = torch.LongTensor(self.negative_emotions).to(config.device)
        # Prior encoding
        self.h_prior = nn.Linear(hidden_dim, hidden_dim)

        self.mu_prior_positive = nn.Linear(hidden_dim, out_dim)
        self.logvar_prior_positive = nn.Linear(hidden_dim, out_dim)
        self.Dense_z_prior_positive = nn.Linear(out_dim, len(self.positive_emotions))

        self.mu_prior_negative = nn.Linear(hidden_dim, out_dim)
        self.logvar_prior_negative = nn.Linear(hidden_dim, out_dim)
        self.Dense_z_prior_negative = nn.Linear(out_dim, len(self.negative_emotions))

        # Posterior encoder
        self.h_posterior_postive = nn.Linear(hidden_dim + posterior_hidden_dim, hidden_dim)
        self.h_posterior_negative = nn.Linear(hidden_dim + posterior_hidden_dim, hidden_dim)

        self.mu_posterior_positive = nn.Linear(hidden_dim, out_dim)
        self.logvar_posterior_positive = nn.Linear(hidden_dim, out_dim)
        self.Dense_z_posterior_positive = nn.Linear(out_dim, len(self.positive_emotions))

        self.mu_posterior_negative = nn.Linear(hidden_dim, out_dim)
        self.logvar_posterior_negative = nn.Linear(hidden_dim, out_dim)
        self.Dense_z_posterior_negative = nn.Linear(out_dim, len(self.negative_emotions))

    def prior(self, x):
        h1 = F.relu(self.h_prior(x))
        mu_positive = self.mu_prior_positive(h1)
        logvar_positive = self.logvar_prior_positive(h1)
        mu_negative = self.mu_prior_negative(h1)
        logvar_negative = self.logvar_prior_negative(h1)
        return mu_positive, logvar_positive, mu_negative, logvar_positive

    def posterior(self, x, e, M_out, M_tilde_out):
        h1_positive = torch.zeros(M_out.shape).to(config.device)
        h1_negative = torch.zeros(M_out.shape).to(config.device)
        for i in range(len(e)):
            if self.is_pos(e[i]):
                h1_positive[i] = M_out[i]
                h1_negative[i] = M_tilde_out[i]
            else:
                h1_positive[i] = M_tilde_out[i]
                h1_negative[i] = M_out[i]
        # Postive
        x_positive = torch.cat([x, h1_positive], dim=-1)
        h1_positive = F.relu(self.h_posterior_postive(x_positive))
        mu_positive = self.mu_posterior_positive(h1_positive)
        logvar_positive = self.logvar_posterior_positive(h1_positive)
        # Negative
        x_negative = torch.cat([x, h1_negative], dim=-1)
        h1_negative = F.relu(self.h_posterior_negative(x_negative))
        mu_negative = self.mu_posterior_negative(h1_negative)
        logvar_negative = self.logvar_posterior_negative(h1_negative)

        return mu_positive, logvar_positive, mu_negative, logvar_positive

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def is_pos(self, e):
        return e in self.positive_emotions

    def forward(self, q_h, e, emb_layer):
        """This method is for evaluation only"""
        x = q_h
        mu_p, logvar_p, mu_n, logvar_n = self.prior(x)

        z_p = self.reparameterize(mu_p, logvar_p)
        E_prob_p = torch.softmax(self.Dense_z_prior_positive(z_p), dim=-1)  # (bs, len(pos))
        emotions_p = (E_prob_p @ emb_layer(self.positive_emotions_t))  # (bs, dim)

        z_n = self.reparameterize(mu_n, logvar_n)
        E_prob_n = torch.softmax(self.Dense_z_prior_negative(z_n), dim=-1)  # (bs, len(neg))
        emotions_n = (E_prob_n @ emb_layer(self.negative_emotions_t))

        emotions_mimic = torch.zeros(emotions_n.shape)
        emotions_non_mimic = torch.zeros(emotions_n.shape)
        for i in range(len(e)):
            if self.is_pos(e[i]):
                emotions_mimic[i] = emotions_p[i]
                emotions_non_mimic[i] = emotions_n[i]
            else:
                emotions_mimic[i] = emotions_n[i]
                emotions_non_mimic[i] = emotions_p[i]

        emotions_mimic = emotions_mimic.to(config.device)
        emotions_non_mimic = emotions_non_mimic.to(config.device)

        return emotions_mimic, emotions_non_mimic, mu_p, logvar_p, mu_n, logvar_n

    def forward_train(self, q_h, e, emb_layer, M_out, M_tilde_out):
        mu_p, logvar_p, mu_n, logvar_n = self.posterior(q_h, e, M_out, M_tilde_out)

        z_p = self.reparameterize(mu_p, logvar_p)
        E_prob_p = torch.softmax(self.Dense_z_prior_positive(z_p), dim=-1)  # (bs, len(pos))
        emotions_p = (E_prob_p @ emb_layer(self.positive_emotions_t))  # (bs, dim)

        z_n = self.reparameterize(mu_n, logvar_n)
        E_prob_n = torch.softmax(self.Dense_z_prior_negative(z_n), dim=-1)  # (bs, len(neg))
        emotions_n = (E_prob_n @ emb_layer(self.negative_emotions_t))

        emotions_mimic = torch.zeros(emotions_n.shape)
        emotions_non_mimic = torch.zeros(emotions_n.shape)
        for i in range(len(e)):
            if self.is_pos(e[i]):
                emotions_mimic[i] = emotions_p[i]
                emotions_non_mimic[i] = emotions_n[i]
            else:
                emotions_mimic[i] = emotions_n[i]
                emotions_non_mimic[i] = emotions_p[i]

        emotions_mimic = emotions_mimic.to(config.device)
        emotions_non_mimic = emotions_non_mimic.to(config.device)

        return emotions_mimic, emotions_non_mimic, mu_p, logvar_p, mu_n, logvar_n

    @staticmethod
    def kl_div(mu_posterior, logvar_posterior, mu_prior=None, logvar_prior=None):
        """
        This code is adapted from:
        https://github.com/ctr4si/A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling/blob/83ca9dd96272d3a38978a1dfa316d06d5d6a7c77/model/utils/probability.py#L20
        """
        one = torch.FloatTensor([1.0])
        if mu_prior == None:
            mu_prior = torch.FloatTensor([0.0])
            logvar_prior = torch.FloatTensor([0.0])

        one = one.to(config.device)
        mu_prior = mu_prior.to(config.device)
        logvar_prior = logvar_prior.to(config.device)
        kl_div = torch.sum(0.5 * (logvar_prior - logvar_posterior + (
                    logvar_posterior.exp() + (mu_posterior - mu_prior).pow(2)) / logvar_prior.exp() - one))
        return kl_div


class EmotionInputEncoder(nn.Module):
    def __init__(self, emb_dim, hidden_size, num_layers, num_heads,
                 total_key_depth, total_value_depth,
                 filter_size, universal, emo_input):

        super(EmotionInputEncoder, self).__init__()
        self.emo_input = emo_input
        if self.emo_input == "self_att":
            self.enc = Encoder(2 * emb_dim, hidden_size, num_layers, num_heads,
                               total_key_depth, total_value_depth,
                               filter_size, universal=universal)
        elif self.emo_input == "cross_att":
            self.enc = Decoder(emb_dim, hidden_size, num_layers, num_heads,
                               total_key_depth, total_value_depth,
                               filter_size, universal=universal)
        else:
            raise ValueError("Invalid attention mode.")

    def forward(self, emotion, encoder_outputs, mask_src):
        if self.emo_input == "self_att":
            repeat_vals = [-1] + [encoder_outputs.shape[1] // emotion.shape[1]] + [-1]
            hidden_state_with_emo = torch.cat([encoder_outputs, emotion.expand(repeat_vals)], dim=2)
            return self.enc(hidden_state_with_emo, mask_src)
        elif self.emo_input == "cross_att":
            return self.enc(encoder_outputs, emotion, (None, mask_src))[0]


class ComplexResDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, universal=False):
        super(ComplexResDecoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if (self.universal):
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length),  # mandatory
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)
        if (self.universal):
            self.dec = ComplexEmoAttentionLayer(*params)
        else:
            self.dec = nn.Sequential(*[ComplexEmoAttentionLayer(*params) for _ in range(num_layers)])

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, m, m_tilt, mask):
        mask_src = mask
        # Add input dropout
        x = self.input_dropout(inputs)
        # Project to hidden size
        x = self.embedding_proj(x)
        if self.universal:
            if config.act:
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.dec, self.timing_signal,
                                                                              self.position_signal, self.num_layers,
                                                                              m, decoding=True)
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x, _, attn_dist, _ = self.dec((x, m, m_tilt, [], mask_src))
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                y, _, attn_dist, _ = self.dec((x, m, m_tilt, [], mask_src))

            y = self.layer_norm(y)
        return y


class ComplexResGate(nn.Module):
    def __init__(self, embedding_size):
        super(ComplexResGate, self).__init__()
        self.fc1 = nn.Linear(2*embedding_size, 2*embedding_size)
        self.fc2 = nn.Linear(2*embedding_size, embedding_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, m, m_tild):
        m_concat = torch.cat((m, m_tild), dim=2)
        x = self.fc1(m_concat)
        z = self.sigmoid(x)
        y = self.fc2(z * m_concat)
        return y


class MIME(nn.Module):
    def __init__(self, vocab, decoder_number, model_file_path=None, is_eval=False, load_optim=False):
        super().__init__()
        self.iter = 0
        self.current_loss = 1000
        self.device = config.device
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab, config.emb_dim, config.PAD_idx, config.pretrain_emb)

        self.encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
                               total_key_depth=config.depth, total_value_depth=config.depth,
                               filter_size=config.filter, universal=config.universal)

        self.decoder_number = decoder_number

        self.decoder = DecoderContextV(config.emb_dim, config.hidden_dim, num_layers=config.hop,
                num_heads=config.heads, total_key_depth=config.depth, total_value_depth=config.depth,
                filter_size=config.filter)

        self.vae_sampler = VAESampling(config.hidden_dim, config.hidden_dim, out_dim=300)

        # outputs m
        self.emotion_input_encoder_1 = EmotionInputEncoder(config.emb_dim, config.hidden_dim, num_layers=config.hop,
                                    num_heads=config.heads, total_key_depth=config.depth, total_value_depth=config.depth,
                                    filter_size=config.filter, universal=config.universal, emo_input=config.emo_input)
        # outputs m~
        self.emotion_input_encoder_2 = EmotionInputEncoder(config.emb_dim, config.hidden_dim, num_layers=config.hop,
                                    num_heads=config.heads, total_key_depth=config.depth, total_value_depth=config.depth,
                                    filter_size=config.filter, universal=config.universal, emo_input=config.emo_input)

        if config.emo_combine == "att":
            self.cdecoder = ComplexResDecoder(config.emb_dim, config.hidden_dim, num_layers=config.hop,
                                     num_heads=config.heads, total_key_depth=config.depth, total_value_depth=config.depth,
                                     filter_size=config.filter, universal=config.universal)

        elif config.emo_combine == "gate":
            self.cdecoder = ComplexResGate(config.emb_dim)

        self.s_weight = nn.Linear(config.hidden_dim, config.emb_dim, bias=False)
        self.decoder_key = nn.Linear(config.hidden_dim, decoder_number, bias=False)

        self.e_weight = nn.Linear(config.emb_dim, config.emb_dim, bias=True)
        self.v = torch.rand(config.emb_dim, requires_grad=True).to(self.device)

        self.generator = Generator(config.hidden_dim, self.vocab_size)
        self.emoji_embedding = nn.Embedding(32, config.emb_dim)

        if config.weight_sharing:
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if config.label_smoothing:
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)

        if config.softmax:
            self.attention_activation = nn.Softmax(dim=1)
        else:
            self.attention_activation = nn.Sigmoid()  # nn.Softmax()

        if config.noam:
            optimizer = torch.optim.Adam(self.parameters(), lr=0, weight_decay=config.weight_decay, betas=(0.9, 0.98), eps=1e-9)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[config.schedule * i for i in range(4)],
                                                             gamma=0.1)
            self.scheduler = NoamOpt(config.hidden_dim, 1, 8000, optimizer, scheduler)
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[config.schedule * i for i in range(4)], gamma=0.1)

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.load_state_dict(state['model'])
            self.iter = state['iter']
            self.current_loss = state['current_loss']
            if load_optim:
                try:
                    self.scheduler.load_state_dict(state['optimizer'])
                except AttributeError:
                    pass

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.positive_emotions = [11, 16, 6, 8, 3, 1, 28, 13, 31, 17, 24, 0, 27]
        self.negative_emotions = [9, 4, 2, 22, 14, 30, 29, 25, 15, 10, 23, 19, 18, 21, 7, 20, 5, 26, 12]

    def save_model(self, running_avg_ppl, iter, f1_g, f1_b, ent_g, ent_b):
        self.iter = iter
        state = {
            'iter': iter,
            'optimizer': self.scheduler.state_dict(),
            'current_loss': running_avg_ppl,
            'model': self.state_dict()
        }
        model_save_path = os.path.join(self.model_dir, 'model_{}_{:.4f}'.format(iter, running_avg_ppl))
        torch.save(state, model_save_path)

    def random_sampling(self, e):
        p = np.random.choice(self.positive_emotions)
        n = np.random.choice(self.negative_emotions)
        if e in self.positive_emotions:
            mimic = p
            mimic_t = n
        else:
            mimic = n
            mimic_t = p
        return mimic, mimic_t

    def train_one_batch(self, batch, iter, train=True):
        enc_batch, cause_batch = get_input_from_batch(batch)
        dec_batch, dec_lengths = get_output_from_batch(batch)
        if config.noam:
            self.scheduler.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        ## encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mak = self.embedding(batch["mask_input"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mak, mask_src)

        q_h = encoder_outputs[:, 0]
        emotions_mimic, emotions_non_mimic, mu_positive_prior, logvar_positive_prior, mu_negative_prior, logvar_negative_prior = \
            self.vae_sampler(q_h, batch['program_label'], self.emoji_embedding)

        m_out = self.emotion_input_encoder_1(emotions_mimic.unsqueeze(1), encoder_outputs, mask_src)
        m_tilde_out = self.emotion_input_encoder_2(emotions_non_mimic.unsqueeze(1), encoder_outputs, mask_src)
        if train:
            emotions_mimic, emotions_non_mimic, mu_positive_posterior, logvar_positive_posterior, mu_negative_posterior, logvar_negative_posterior = \
                self.vae_sampler.forward_train(q_h, batch['program_label'], self.emoji_embedding,
                                               M_out=m_out.mean(dim=1), M_tilde_out=m_tilde_out.mean(dim=1))
            KLLoss_positive = self.vae_sampler.kl_div(mu_positive_posterior, logvar_positive_posterior,
                                                      mu_positive_prior, logvar_positive_prior)
            KLLoss_negative = self.vae_sampler.kl_div(mu_negative_posterior, logvar_negative_posterior,
                                                      mu_negative_prior, logvar_negative_prior)
            KLLoss = KLLoss_positive + KLLoss_negative
        else:
            KLLoss_positive = self.vae_sampler.kl_div(mu_positive_prior, logvar_positive_prior)
            KLLoss_negative = self.vae_sampler.kl_div(mu_negative_prior, logvar_negative_prior)
            KLLoss = KLLoss_positive + KLLoss_negative

        if config.emo_combine == "att":
            v = self.cdecoder(encoder_outputs, m_out, m_tilde_out, mask_src)
        elif config.emo_combine == "gate":
            v = self.cdecoder(m_out, m_tilde_out)

        x = self.s_weight(q_h)

        # method2: E (W@c)
        logit_prob = torch.matmul(x, self.emoji_embedding.weight.transpose(0, 1))  # shape (b_size, 32)

        # Decode
        sos_token = torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1).to(self.device)
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), 1)

        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)

        pre_logit, attn_dist = self.decoder(self.embedding(dec_batch_shift), v, v, (mask_src, mask_trg))

        ## compute output dist
        logit = self.generator(pre_logit)

        if (train and config.schedule > 10):
            if (random.uniform(0, 1) <= (0.0001 + (1 - 0.0001) * math.exp(-1. * iter / config.schedule))):
                config.oracle = True
            else:
                config.oracle = False

        if config.softmax:
            program_label = torch.LongTensor(batch['program_label']).to(self.device)

            if config.emo_combine == 'gate':
                L1_loss = nn.CrossEntropyLoss()(logit_prob, program_label)
                loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)),
                                      dec_batch.contiguous().view(-1)) + KLLoss + L1_loss
            else:
                L1_loss = nn.CrossEntropyLoss()(logit_prob, torch.LongTensor(batch['program_label']).to(self.device))
                loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)),
                                      dec_batch.contiguous().view(-1)) + KLLoss + L1_loss

            loss_bce_program = nn.CrossEntropyLoss()(logit_prob, program_label).item()
        else:
            loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)),
                                  dec_batch.contiguous().view(-1)) + nn.BCEWithLogitsLoss()(logit_prob,
                                                                                            torch.FloatTensor(batch['target_program']).to(self.device))
            loss_bce_program = nn.BCEWithLogitsLoss()(logit_prob,
                                                      torch.FloatTensor(batch['target_program']).to(self.device)).item()
        pred_program = np.argmax(logit_prob.detach().cpu().numpy(), axis=1)
        program_acc = accuracy_score(batch["program_label"], pred_program)

        if config.label_smoothing:
            loss_ppl = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)),
                                          dec_batch.contiguous().view(-1)).item()

        if train:
            loss.backward()
            self.scheduler.step()

        if config.label_smoothing:
            return loss_ppl, math.exp(min(loss_ppl, 100)), loss_bce_program, program_acc
        else:
            return loss.item(), math.exp(min(loss.item(), 100)), loss_bce_program, program_acc

    def compute_act_loss(self, module):
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t, dim=1) / p_t.size(1)) / p_t.size(0)
        loss = config.act_loss_weight * avg_p_t.item()
        return loss

    def decoder_greedy(self, batch, max_dec_step=30, emotion_classifier='built_in'):
        enc_batch, cause_batch = get_input_from_batch(batch)

        ## encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mak = self.embedding(batch["mask_input"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mak, mask_src)

        q_h = encoder_outputs[:, 0]
        x = self.s_weight(q_h)
        logit_prob = torch.matmul(x, self.emoji_embedding.weight.transpose(0, 1))
        emo_pred = torch.argmax(logit_prob, dim=-1)

        if emotion_classifier == "vader":
            context_emo = [self.positive_emotions[0] if d['compound'] > 0 else self.negative_emotions[0] for d in batch['context_emotion_scores']]
            context_emo = torch.Tensor(context_emo).to(self.device)
            emotions_mimic, emotions_non_mimic, mu_p, logvar_p, mu_n, logvar_n = self.vae_sampler(q_h, context_emo, self.emoji_embedding)
        elif emotion_classifier == None:
            emotions_mimic, emotions_non_mimic, mu_p, logvar_p, mu_n, logvar_n = self.vae_sampler(q_h, batch['program_label'], self.emoji_embedding)
        elif emotion_classifier == "built_in":
            emotions_mimic, emotions_non_mimic, mu_p, logvar_p, mu_n, logvar_n = self.vae_sampler(q_h, emo_pred, self.emoji_embedding)

        m_out = self.emotion_input_encoder_1(emotions_mimic.unsqueeze(1), encoder_outputs, mask_src)
        m_tilde_out = self.emotion_input_encoder_2(emotions_non_mimic.unsqueeze(1), encoder_outputs, mask_src)

        if config.emo_combine == "att":
            v = self.cdecoder(encoder_outputs, m_out, m_tilde_out, mask_src)
        elif config.emo_combine == "gate":
            v = self.cdecoder(m_out, m_tilde_out)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(self.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            out, attn_dist = self.decoder(self.embedding(ys), v, v, (mask_src, mask_trg))
            logit = self.generator(out)
            filtered_logit = top_k_top_p_filtering(logit[:, -1], top_k=3, top_p=0, filter_value=-float('Inf'))
            # Sample from the filtered distribution
            next_word = torch.multinomial(F.softmax(filtered_logit, dim=-1), 1).squeeze()
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in
                                  next_word.view(-1)])
            next_word = next_word.data.item()
            ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).to(self.device)], dim=1).to(self.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>':
                    break
                else:
                    st += e + ' '
            sent.append(st)
        return sent


    def decoder_topk(self, batch, max_dec_step=30, emotion_classifier='built_in'):
        enc_batch, cause_batch = get_input_from_batch(batch)

        ## Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)

        emb_mask = self.embedding(batch["mask_input"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, mask_src)

        q_h = encoder_outputs[:, 0]

        x = self.s_weight(q_h)
        # method 2
        logit_prob = torch.matmul(x, self.emoji_embedding.weight.transpose(0, 1))

        if emotion_classifier == "vader":
            context_emo = [self.positive_emotions[0] if d['compound'] > 0 else self.negative_emotions[0] for d in
                           batch['context_emotion_scores']]
            context_emo = torch.Tensor(context_emo).to(self.device)
            emotions_mimic, emotions_non_mimic, mu_p, logvar_p, mu_n, logvar_n = self.vae_sampler(q_h, context_emo,
                                                                                                  self.emoji_embedding)
        elif emotion_classifier == None:
            emotions_mimic, emotions_non_mimic, mu_p, logvar_p, mu_n, logvar_n = self.vae_sampler(q_h, batch[
                'program_label'], self.emoji_embedding)
        elif emotion_classifier == "built_in":
            emo_pred = torch.argmax(logit_prob, dim=-1)
            emotions_mimic, emotions_non_mimic, mu_p, logvar_p, mu_n, logvar_n = self.vae_sampler(q_h, emo_pred,
                                                                                                  self.emoji_embedding)

        m_out = self.emotion_input_encoder_1(emotions_mimic.unsqueeze(1), encoder_outputs, mask_src)
        m_tilde_out = self.emotion_input_encoder_2(emotions_non_mimic.unsqueeze(1), encoder_outputs, mask_src)

        if config.emo_combine == "att":
            v = self.cdecoder(encoder_outputs, m_out, m_tilde_out, mask_src)
        elif config.emo_combine == "gate":
            v = self.cdecoder(m_out, m_tilde_out)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(self.device)

        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            out, attn_dist = self.decoder(self.embedding(ys), v, v, (mask_src, mask_trg))

            logit = self.generator(out)
            filtered_logit = top_k_top_p_filtering(logit[:, -1], top_k=3, top_p=0, filter_value=-float('Inf'))
            # Sample from the filtered distribution
            next_word = torch.multinomial(F.softmax(filtered_logit, dim=-1), 1).squeeze()
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in
                                  next_word.view(-1)])
            next_word = next_word.data.item()

            ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).to(self.device)], dim=1).to(self.device)

            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>':
                    break
                else:
                    st += e + ' '
            sent.append(st)
        return sent

class ACT_basic(nn.Module):
    """

    """
    def __init__(self,hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size,1)
        self.p.bias.data.fill_(1)
        self.threshold = 1 - 0.1

    def forward(self, state, inputs, fn, time_enc, pos_enc, max_hop, encoder_output=None, decoding=False):
        # init_hdd
        ## [B, S]
        halting_probability = torch.zeros(inputs.shape[0],inputs.shape[1]).to(config.device)
        ## [B, S]
        remainders = torch.zeros(inputs.shape[0],inputs.shape[1]).to(config.device)
        ## [B, S]
        n_updates = torch.zeros(inputs.shape[0],inputs.shape[1]).to(config.device)
        ## [B, S, HDD]
        previous_state = torch.zeros_like(inputs).to(config.device)

        step = 0
        # for l in range(self.num_layers):
        while ((halting_probability<self.threshold) & (n_updates < max_hop)).byte().any():
            # as long as there is a True value, the loop continues
            # Add timing signal
            state = state + time_enc[:, :inputs.shape[1], :].type_as(inputs.data)
            state = state + pos_enc[:, step, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)

            p = self.sigma(self.p(state)).squeeze(-1) # (1, 1)
            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            if decoding:
                state, _, attention_weight = fn((state,encoder_output,[]))
            else:
                # apply transformation on the state
                state = fn(state)

            # update running part in the weighted state and keep the rest
            previous_state = ((state * update_weights.unsqueeze(-1)) + (previous_state * (1 - update_weights.unsqueeze(-1))))
            if decoding:
                if(step==0):  previous_att_weight = torch.zeros_like(attention_weight).to(config.device)     ## [B, S, src_size]
                previous_att_weight = ((attention_weight * update_weights.unsqueeze(-1)) + (previous_att_weight * (1 - update_weights.unsqueeze(-1))))
            ## previous_state is actually the new_state at end of hte loop
            ## to save a line I assigned to previous_state so in the next
            ## iteration is correct. Notice that indeed we return previous_state
            step+=1

        if decoding:
            return previous_state, previous_att_weight, (remainders,n_updates)
        else:
            return previous_state, (remainders,n_updates)

