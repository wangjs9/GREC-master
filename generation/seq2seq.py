import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from models.common_layer import share_embedding, NoamOpt, LabelSmoothing, get_input_from_batch, get_output_from_batch, \
    _get_attn_subsequent_mask
import config
import pprint
pp = pprint.PrettyPrinter(indent=1)
import os
from sklearn.metrics import accuracy_score

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


class Encoder(nn.Module):

    def __init__(self, emb_dim, hidden_size, input_dropout=0.2):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(emb_dim, hidden_size, batch_first=True, bidirectional=True)
        self.input_dropout = nn.Dropout(input_dropout)
        self.fc = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, inputs):
        inputs = self.input_dropout(inputs)
        out, hid = self.rnn(inputs)
        hid = torch.cat([hid[-2], hid[-1]], dim=1)
        hid = nn.Tanh()(self.fc(hid)).unsqueeze(0)

        return out, hid


class Attention(nn.Module):

    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attention, self).__init__()
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.linear_in = nn.Linear(enc_hidden_size * 2, dec_hidden_size, bias=False)
        self.linear_out = nn.Linear(enc_hidden_size * 2 + dec_hidden_size, dec_hidden_size)

    def forward(self, output, context, mask):
        batch_size = output.size(0)
        output_len = output.size(1)
        input_len = context.size(1)

        context_in = self.linear_in(context.view(batch_size * input_len, -1)).view(
            batch_size, input_len, -1)  # batch_size, context_len, dec_hidden_size

        # context_in.transpose(1,2): batch_size, dec_hidden_size, context_len
        # output: batch_size, output_len, dec_hidden_size
        attn = torch.bmm(output, context_in.transpose(1, 2))
        # batch_size, output_len, context_len

        attn.data.masked_fill(mask, -1e6)

        attn = F.softmax(attn, dim=2)
        # batch_size, output_len, context_len

        context = torch.bmm(attn, context)
        # batch_size, output_len, enc_hidden_size

        output = torch.cat((context, output), dim=2)  # batch_size, output_len, hidden_size*2

        output = output.view(batch_size * output_len, -1)
        output = torch.tanh(self.linear_out(output))
        output = output.view(batch_size, output_len, -1)
        return output, attn


class Decoder(nn.Module):

    def __init__(self, emb_dim, hidden_size, max_length=1000, input_dropout=0.2):
        super(Decoder, self).__init__()
        self.attention = Attention(hidden_size, hidden_size)
        self.rnn = nn.GRU(emb_dim, hidden_size, batch_first=True)
        self.mask = _get_attn_subsequent_mask(max_length)
        self.dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)], 0)
        out, hid = self.rnn(inputs)
        y, attn_dist = self.attention(out, encoder_output, dec_mask)
        return y, attn_dist

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(self, x):
        logit = self.proj(x)
        return F.log_softmax(logit, dim=-1)


class S2S(nn.Module):

    def __init__(self, vocab, decoder_number, model_file_path=None, load_optim=False):
        super(S2S, self).__init__()
        self.iter = 0
        self.current_loss = 1000
        self.device = config.device
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab, config.emb_dim, config.PAD_idx, config.pretrain_emb)
        self.encoder = Encoder(config.emb_dim, config.hidden_dim)
        self.decoder_number = decoder_number
        self.decoder = Decoder(config.emb_dim, config.hidden_dim)

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
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[config.schedule * i for i in range(4)],
                                                             gamma=0.1)
            self.scheduler = NoamOpt(config.hidden_dim, 1, 8000, optimizer, scheduler)
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=[config.schedule * i for i in range(4)],
                                                                  gamma=0.1)

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.iter = state['iter']
            self.current_loss = state['current_loss']
            self.load_state_dict(state['model'])
            if load_optim:
                try:
                    self.scheduler.load_state_dict(state['optimizer'])
                except AttributeError:
                    pass

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

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

        sos_token = torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1).to(config.device)
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), 1)  # make the first token of sentence be SOS

        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)
        pre_logit, attn_dist = self.decoder(self.embedding(dec_batch_shift), encoder_outputs, (mask_src, mask_trg))

        logit = self.generator(pre_logit)

        loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))

        loss_bce_program, program_acc = 0, 0
        # multi-task
        if config.emo_multitask:
            # add the loss function of label prediction
            q_h = encoder_outputs[:, 0]  # the first token of the sentence CLS, shape: (batch_size, 1, hidden_size)
            logit_prob = self.decoder_key(q_h).to('cuda')  # (batch_size, 1, decoder_num)
            loss += nn.CrossEntropyLoss()(logit_prob, torch.LongTensor(batch['program_label']).cuda())
            loss_bce_program = nn.CrossEntropyLoss()(logit_prob, torch.LongTensor(batch['program_label']).cuda()).item()
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
        enc_batch, cause_batch = get_input_from_batch(batch)

        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["mask_input"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, mask_src)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):

            out, attn_dist = self.decoder(self.embedding(ys), encoder_outputs, (mask_src, mask_trg))

            prob = self.generator(out)
            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in
                                  next_word.view(-1)])
            next_word = next_word.data[0]

            ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)], dim=1).to(config.device)
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




