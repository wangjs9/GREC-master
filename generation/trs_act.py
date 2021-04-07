import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from models.common_layer import EncoderLayer, DecoderLayer, LayerNorm , \
    _gen_bias_mask ,_gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, \
    _get_attn_subsequent_mask,  get_input_from_batch, get_graph_from_batch, get_output_from_batch
import config
import pprint
pp = pprint.PrettyPrinter(indent=1)
import os
from torch_scatter import scatter_max, scatter_mean, scatter_add
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

        if (config.act):
            self.act_fn = ACT_basic(hidden_size)
            self.remainders = None
            self.n_updates = None

    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs) # (batch_size, seq_len, embed_dim)
        # Project to hidden size
        x = self.embedding_proj(x) # (batch_size, seq_len, hidden_size)

        if self.universal:
            if config.act:
                x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal,
                                                                   self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x = self.enc(x, mask=mask)
                    # x = torch.mul(self.enc(x, mask=mask), cazprob + 1)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)

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
                    x, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))

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


class Effect(nn.Module):
    def __init__(self, vocab, decoder_number, act_num=8, model_file_path=None, load_optim=False):
        """
        vocab: a Lang type data, which is defined in data_reader.py
        decoder_number: the number of classes
        """
        super(Effect, self).__init__()
        self.iter = 0
        self.current_loss = 1000
        self.device = config.device
        self.vocab = vocab
        self.vocab_size = vocab.n_words
        self.act_num = act_num

        self.embedding = share_embedding(self.vocab, config.emb_dim, config.PAD_idx, config.pretrain_emb)

        self.encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
                               total_key_depth=config.depth, total_value_depth=config.depth,
                               filter_size=config.filter, universal=config.universal)

        self.decoder = Decoder(config.emb_dim, hidden_size=config.hidden_dim, num_layers=config.hop,
                               num_heads=config.heads,
                               total_key_depth=config.depth, total_value_depth=config.depth,
                               filter_size=config.filter)

        self.decoder_key = nn.Linear(config.hidden_dim, decoder_number, bias=False)
        self.act_decoder = nn.Linear(config.hidden_dim, act_num, bias=False)
        self.generator = Generator(config.hidden_dim, self.vocab_size)

        if config.weight_sharing:
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if config.label_smoothing:
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)

        if config.noam:
            optimizer = torch.optim.Adam(self.parameters(), lr=0, weight_decay=config.weight_decay, betas=(0.9, 0.98),
                                         eps=1e-9)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[config.schedule*i for i in range(4)], gamma=0.1)
            self.scheduler = NoamOpt(config.hidden_dim, 1, 8000, optimizer, scheduler)
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[config.schedule*i for i in range(4)], gamma=0.1)

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.iter = state['iter']
            self.current_loss = state['current_loss']
            self.embedding.load_state_dict(state['embedding_dict'])
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'])
            self.generator.load_state_dict(state['generator_dict'])
            self.decoder_key.load_state_dict(state['decoder_key_state_dict'])
            if load_optim:
                try:
                    self.scheduler.load_state_dict(state['optimizer'])
                except AttributeError:
                    pass

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, running_avg_ppl, iter, f1_g, f1_b, ent_g, ent_b):
        self.iter = iter
        state = {
            'iter': self.iter,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'act_decoder_dict': self.decoder_key.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'decoder_key_state_dict': self.decoder_key.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            'optimizer': self.scheduler.state_dict(),
            'current_loss': running_avg_ppl
        }

        model_save_path = os.path.join(self.model_dir, 'model_{}_{:.4f}'.format(
            iter, running_avg_ppl))
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def train_one_batch(self, batch, iter, train=True):
        enc_batch, _ = get_input_from_batch(batch)
        dec_batch, dec_lengths, _ = get_output_from_batch(batch)
        target_act = batch["target_act"].to(self.device)
        if config.noam:
            self.scheduler.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        ## encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mak = self.embedding(batch["mask_input"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mak,
                                       mask_src)
        q_h = encoder_outputs[:, 0]  # the first token of the sentence CLS, shape: (batch_size, 1, hidden_size)

        ## decode
        sos_token = torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1).to(self.device)
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), 1)
        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)
        dec_input = self.embedding(dec_batch_shift)

        ## logit and graph processing
        pre_logit, attn_dist = self.decoder(dec_input, encoder_outputs, (mask_src, mask_trg))
        act_distribute = F.log_softmax(self.act_decoder(pre_logit))
        logit = self.generator(pre_logit).to(self.device)

        loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
        # multi-task

        logit_prob = self.decoder_key(q_h).to(self.device)  # (batch_size, 1, decoder_num)
        loss += nn.CrossEntropyLoss()(logit_prob, torch.LongTensor(batch['program_label']).to(self.device))
        loss += nn.CrossEntropyLoss(ignore_index=-1)(act_distribute.view(-1, self.act_num), target_act.view(-1, ))
        loss_bce_program = nn.CrossEntropyLoss()(logit_prob, torch.LongTensor(batch['program_label']).to(self.device)).item()
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

    def decoder_greedy(self, batch, max_dec_step=30):
        enc_batch, _ = get_input_from_batch(batch)

        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["mask_input"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask,
                                       mask_src)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(self.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            dec_input = self.embedding(ys)
            out, attn_dist = self.decoder(dec_input, encoder_outputs, (mask_src, mask_trg))
            prob = self.generator(out)
            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in
                                  next_word.view(-1)])
            next_word = next_word.data[0]

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

