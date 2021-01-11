import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from models.analysis_common_layer import EncoderLayer, DecoderLayer, LayerNorm , \
    _gen_bias_mask ,_gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, \
    _get_attn_subsequent_mask,  get_input_from_batch, get_graph_from_batch, get_output_from_batch, \
    top_k_top_p_filtering, PositionwiseFeedForward, gaussian_kld
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

        if (self.universal):
            if (config.act):
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

    def forward(self, x, attn_dist=None, enc_batch_extend_vocab=None,
                temp=1, beam_search=False, attn_dist_db=None):

        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)

        if(config.pointer_gen):
            vocab_dist = F.softmax(logit/temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist/temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab.unsqueeze(1)]*x.size(1),1) ## extend for all seq
            if(beam_search):
                enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab_[0].unsqueeze(0)]*x.size(0),0) ## extend for all seq
            logit = torch.log(vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_))
            return logit
        else:
            return F.log_softmax(logit,dim=-1)

class GLSTM(nn.Module):
    def __init__(self, embedding):
        super(GLSTM, self).__init__()
        self.embedding = embedding
        self.device = config.device
        self.hop_num = config.hop_num
        self.relation_embd = nn.Embedding(47*2, config.emb_dim)
        self.W_s = nn.ModuleList(
            [nn.Linear(config.emb_dim, config.emb_dim, bias=False) for _ in range(config.hop_num)])
        self.W_n = nn.ModuleList(
            [nn.Linear(config.emb_dim, config.emb_dim, bias=False) for _ in range(config.hop_num)])
        self.W_r = nn.ModuleList(
            [nn.Linear(config.emb_dim, config.emb_dim, bias=False) for _ in range(config.hop_num)])
        self.triple_linear = nn.Linear(config.emb_dim * 3, config.emb_dim, bias=False)
        self.lstm = GCNLSTM()
        self.gate_linear = nn.Linear(config.emb_dim, 1)

    def multi_layer_gcn(self, concept_hidden, relation_hidden, head, tail, triple_label,
                             layer_number=2):
        for i in range(layer_number):
            concept_hidden, relation_hidden = self.gcn(concept_hidden, relation_hidden, head, tail,
                                                            triple_label, i)
        return concept_hidden, relation_hidden

    def gcn(self, concept_hidden, relation_hidden, head, tail, triple_label, layer_idx):
        # shape:
        # concept_hidden: (batch_size, max_mem_size, hidden_size)
        # head: (batch_size, 5, max_trp_size)
        batch_size = head.size(0)
        max_trp_size = head.size(1)
        max_mem_size = concept_hidden.size(1)
        hidden_size = concept_hidden.size(2)

        update_node = torch.zeros_like(concept_hidden).to(self.device).float() # (batch_size, max_mem_size, hidden_size)
        count = torch.ones_like(head).to(self.device).masked_fill_(triple_label == -1, 0).float() # (batch_size, max_trp_size)
        count_out = torch.zeros(batch_size, max_mem_size).to(head.device).float() # (batch_size, max_mem_size)

        o = concept_hidden.gather(1, head.unsqueeze(2).expand(batch_size, max_trp_size, hidden_size)) # (batch_size, max_trp_size, hidden_size)
        o = o.masked_fill(triple_label.unsqueeze(2) == -1, 0)
        scatter_add(o, tail, dim=1, out=update_node)
        scatter_add(- relation_hidden.masked_fill(triple_label.unsqueeze(2) == -1, 0), tail, dim=1, out=update_node)
        scatter_add(count, tail, dim=1, out=count_out)

        o = concept_hidden.gather(1, tail.unsqueeze(2).expand(batch_size, max_trp_size, hidden_size))
        o = o.masked_fill(triple_label.unsqueeze(2) == -1, 0)
        scatter_add(o, head, dim=1, out=update_node)
        scatter_add(- relation_hidden.masked_fill(triple_label.unsqueeze(2) == -1, 0), head, dim=1, out=update_node)
        scatter_add(count, head, dim=1, out=count_out)

        update_node = self.W_s[layer_idx](concept_hidden) + self.W_n[layer_idx](update_node) / count_out.clamp(
            min=1).unsqueeze(2)
        update_node = nn.ReLU()(update_node)

        return update_node, self.W_r[layer_idx](relation_hidden)

    def multi_hop(self, triple_prob, distance, head, tail, concept_label, triple_label, gamma=0.8, iteration=3,
                       method="avg"):
        '''
        triple_prob: bsz x L x mem_t
        distance: bsz x mem
        head, tail: bsz x mem_t
        concept_label: bsz x mem
        triple_label: bsz x mem_t

        Init binary vector with source concept == 1 and others 0
        expand to size: bsz x L x mem
        '''
        concept_probs = []

        cpt_size = (triple_prob.size(0), triple_prob.size(1), distance.size(1))
        init_mask = torch.zeros_like(distance).unsqueeze(1).expand(*cpt_size).to(distance.device).float()
        init_mask.masked_fill_((distance == 0).unsqueeze(1), 1)
        final_mask = init_mask.clone()

        init_mask.masked_fill_((concept_label == -1).unsqueeze(1), 0)
        concept_probs.append(init_mask)

        head = head.unsqueeze(1).expand(triple_prob.size(0), triple_prob.size(1), -1)
        tail = tail.unsqueeze(1).expand(triple_prob.size(0), triple_prob.size(1), -1)

        for step in range(iteration):
            '''
            Calculate triple head score
            '''
            node_score = concept_probs[-1]
            triple_head_score = node_score.gather(2, head)
            triple_head_score.masked_fill_((triple_label == -1).unsqueeze(1), 0)
            '''
            Method: 
                - avg:
                    s(v) = Avg_{u \in N(v)} gamma * s(u) + R(u->v) 
                - max: 
                    s(v) = max_{u \in N(v)} gamma * s(u) + R(u->v)
            '''
            update_value = triple_head_score * gamma + triple_prob
            out = torch.zeros_like(node_score).to(node_score.device).float()
            if method == "max":
                scatter_max(update_value, tail, dim=-1, out=out)
            elif method == "avg":
                scatter_mean(update_value, tail, dim=-1, out=out)
            out.masked_fill_((concept_label == -1).unsqueeze(1), 0)

            concept_probs.append(out)
        print(concept_probs[2].size())

        index = int(concept_probs[2].size(1))
        with open('text_{}.txt'.format(index), 'w') as f:
            for i in concept_probs[2][0][-1]:
                f.write(str(float(i)))
                f.write('\n')
        '''
        Natural decay of concept that is multi-hop away from source
        '''
        total_concept_prob = final_mask * -1e5
        for prob in concept_probs[1:]:
            total_concept_prob += prob
        # bsz x L x mem

        return total_concept_prob

    def comp_cause(self, concept_ids, relation, head, tail, triple_label):
        self.batch_size = concept_ids.size(0)
        self.graph_num = concept_ids.size(1)

        ## reshape all matrixes
        concept_ids = concept_ids.reshape(-1, concept_ids.size(2))
        relation = relation.reshape(-1, relation.size(2))
        head = head.reshape(-1, head.size(2))
        tail = tail.reshape(-1, tail.size(2))
        triple_label = triple_label.reshape(-1, triple_label.size(2))

        ## calculate graph
        memory = self.embedding(concept_ids)
        rel_repr = self.embedding(relation)
        node_repr, rel_repr = self.multi_layer_gcn(memory, rel_repr, head, tail, triple_label,
                                                        layer_number=2)
        head_repr = torch.gather(node_repr, 1,
                                 head.unsqueeze(-1).expand(node_repr.size(0), head.size(1), node_repr.size(-1)))
        tail_repr = torch.gather(node_repr, 1,
                                 tail.unsqueeze(-1).expand(node_repr.size(0), tail.size(1), node_repr.size(-1)))
        triple_repr = torch.cat((head_repr, rel_repr, tail_repr), dim=-1)
        triple_repr = self.triple_linear(triple_repr)

        encoded_cause = self.lstm(triple_repr.reshape(self.batch_size, self.graph_num, -1, triple_repr.size(-1)))

        assert (not torch.isnan(triple_repr).any().item())

        return triple_repr, encoded_cause

    def comp_pointer(self, hidden_state, concept_label, distance, head, tail, triple_repr, triple_label, vocab_map, map_mask):
        concept_label = concept_label.reshape(-1, concept_label.size(2))
        distance = distance.reshape(-1, distance.size(2))
        head = head.reshape(-1, head.size(2))
        tail = tail.reshape(-1, tail.size(2))
        triple_label = triple_label.reshape(-1, triple_label.size(2))
        new_hidden_state = hidden_state.unsqueeze(1).expand(-1, self.graph_num, -1, -1).reshape(self.batch_size * self.graph_num, hidden_state.size(1), hidden_state.size(2))
        triple_logits = torch.matmul(new_hidden_state, triple_repr.transpose(1, 2))
        triple_prob = nn.Sigmoid()(triple_logits)
        triple_prob = triple_prob.masked_fill((triple_label == -1).unsqueeze(1), 0)

        cpt_probs = self.multi_hop(triple_prob, distance, head, tail, concept_label, triple_label, config.hop_num)
        cpt_probs = cpt_probs.reshape(self.batch_size, self.graph_num, -1, cpt_probs.size(-1)) # bsz x graph_num x L x mem
        # cpt_probs = cpt_probs.transpose(2, 1).reshape(batch_size, -1, graph_num * cpt_probs.size(-1))
        cpt_probs = F.log_softmax(cpt_probs, dim=-1)
        cpt_probs_vocab = cpt_probs.gather(-1, vocab_map.unsqueeze(2).expand(cpt_probs.size(0),
                        cpt_probs.size(1), cpt_probs.size(2), -1))
        cpt_probs_vocab = torch.sum(cpt_probs_vocab, dim=1)
        cpt_probs_vocab.masked_fill_((map_mask == 0).unsqueeze(1), 0)
        # bsz x graph_num x L x vocab

        gate = F.log_softmax(self.gate_linear(hidden_state), dim=-1)
        # bsz x L x 1

        return gate, cpt_probs_vocab

class GCNLSTM(nn.Module):
    def __init__(self, embed_size=config.emb_dim):
        super(GCNLSTM, self).__init__()
        self.linear = nn.Linear(embed_size,embed_size, bias=False)
        self.cause_lstm = nn.GRU(embed_size, config.hidden_dim, batch_first=True, bidirectional=True)
        self.cause_linear = nn.Linear(config.hidden_dim * 2, config.hidden_dim, bias=False)

    def forward(self, cause_batch):
        cause_batch = torch.sum(cause_batch, dim=-2) # batch_size, graph_num, hidden_size
        cause_batch = self.linear(cause_batch)
        _, cause_hid = self.cause_lstm(cause_batch)
        cause_hid = torch.cat((cause_hid[-1], cause_hid[-2]), dim=-1) # batch_size, hidden_size * 2
        encoded_cause = self.cause_linear(cause_hid)  # batch_size, hidden_size
        encoded_cause = torch.tanh(encoded_cause)
        return encoded_cause

class MultiHopCause(nn.Module):
    def __init__(self, vocab, decoder_number, model_file_path=None, load_optim=False):
        """
        vocab: a Lang type data, which is defined in data_reader.py
        decoder_number: the number of classes
        """
        super(MultiHopCause, self).__init__()
        self.iter = 0
        self.current_loss = 1000
        self.device = config.device
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)
        posembedding = torch.FloatTensor(np.load(config.posembedding_path, allow_pickle=True))
        self.causeposembeding = nn.Embedding.from_pretrained(posembedding, freeze=True)

        self.glstm = GLSTM(self.embedding)

        self.encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
                               total_key_depth=config.depth, total_value_depth=config.depth,
                               filter_size=config.filter, universal=config.universal)

        self.decoder = Decoder(config.emb_dim, hidden_size=config.hidden_dim, num_layers=config.hop,
                               num_heads=config.heads,
                               total_key_depth=config.depth, total_value_depth=config.depth,
                               filter_size=config.filter)

        self.decoder_key = nn.Linear(config.hidden_dim, decoder_number, bias=False)
        self.generator = Generator(config.hidden_dim, self.vocab_size)

        if config.weight_sharing:
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if (config.label_smoothing):
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)

        if (config.noam):
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
            self.glstm.load_state_dict(state['cause_encoder_dict'])
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
            'cause_encoder_dict': self.glstm.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'decoder_key_state_dict': self.decoder_key.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            'optimizer': self.scheduler.state_dict(),
            'current_loss': running_avg_ppl
        }
        # model_save_path = os.path.join(self.model_dir, 'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(
        #     iter, running_avg_ppl, f1_g, f1_b, ent_g, ent_b))
        model_save_path = os.path.join(self.model_dir, 'model_{}_{:.4f}'.format(
            iter, running_avg_ppl))
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def train_one_batch(self, batch, train=True):
        enc_batch, cause_batch, enc_batch_extend_vocab = get_input_from_batch(batch)
        graphs, graph_num = get_graph_from_batch(batch)
        dec_batch, dec_lengths, _ = get_output_from_batch(batch)
        if (config.noam):
            self.scheduler.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        ## encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mak = self.embedding(batch["mask_input"])
        causepos = self.causeposembeding(batch["causepos"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mak + causepos,
                                       mask_src)

        ## graph processing
        if torch.sum(graph_num):
            concept_ids, concept_label, distance, relation, head, tail, triple_label, vocab_map, map_mask = graphs
            triple_repr, cause_repr = self.glstm.comp_cause(concept_ids, relation, head, tail, triple_label)

        ## decode
        sos_token = torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1).to(self.device)
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), 1)
        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)
        dec_input = self.embedding(dec_batch_shift)

        if torch.sum(graph_num):
            dec_input[:, 0] = dec_input[:, 0] + cause_repr

        ## logit
        pre_logit, attn_dist = self.decoder(dec_input, encoder_outputs, (mask_src, mask_trg))
        logit = self.generator(pre_logit, attn_dist, enc_batch_extend_vocab if config.pointer_gen else None,
                               attn_dist_db=None)
        if torch.sum(graph_num):
            gate, cpt_probs_vocab = self.glstm.comp_pointer(pre_logit, concept_label, distance, head, tail, triple_repr,
                                triple_label, vocab_map, map_mask)
            logit = logit * (1 - gate) + gate * cpt_probs_vocab

        loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
        loss_bce_program, loss_bce_caz, program_acc = 0, 0, 0

        # multi-task
        if config.emo_multitask:
            # add the loss function of label prediction
            # q_h = torch.mean(encoder_outputs,dim=1)
            q_h = encoder_outputs[:, 0]  # the first token of the sentence CLS, shape: (batch_size, 1, hidden_size)
            logit_prob = self.decoder_key(q_h).to(self.device)  # (batch_size, 1, decoder_num)
            loss += nn.CrossEntropyLoss()(logit_prob, torch.LongTensor(batch['program_label']).cuda())

            loss_bce_program = nn.CrossEntropyLoss()(logit_prob, torch.LongTensor(batch['program_label']).cuda()).item()
            pred_program = np.argmax(logit_prob.detach().cpu().numpy(), axis=1)
            program_acc = accuracy_score(batch["program_label"], pred_program)

        if (config.label_smoothing):
            loss_ppl = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)),
                                          dec_batch.contiguous().view(-1)).item()

        if (train):
            loss.backward()
            self.scheduler.step()
        if (config.label_smoothing):
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
        enc_batch, cause_batch, enc_batch_extend_vocab = get_input_from_batch(batch)
        graphs, graph_num = get_graph_from_batch(batch)

        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["mask_input"])
        causepos = self.causeposembeding(batch["causepos"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask + causepos,
                                       mask_src)

        if torch.sum(graph_num):
            concept_ids, concept_label, distance, relation, head, tail, triple_label, vocab_map, map_mask = graphs
            ## graph_encoder
            triple_repr, cause_repr = self.glstm.comp_cause(concept_ids, relation, head, tail, triple_label)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(self.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            dec_input = self.embedding(ys)
            if torch.sum(graph_num):
                dec_input[:, 0] = dec_input[:, 0] + cause_repr

            out, attn_dist = self.decoder(dec_input, encoder_outputs, (mask_src, mask_trg))

            prob = self.generator(out, attn_dist, enc_batch_extend_vocab, attn_dist_db=None)

            if torch.sum(graph_num):
                gate, cpt_probs_vocab = self.glstm.comp_pointer(out, concept_label, distance, head, tail, triple_repr,
                                                            triple_label, vocab_map, map_mask)
                prob = prob * (1 - gate) + gate * cpt_probs_vocab
            # logit = F.log_softmax(logit,dim=-1) #fix the name later
            # filtered_logit = top_k_top_p_filtering(logit[:, -1], top_k=0, top_p=0, filter_value=-float('Inf'))
            # Sample from the filtered distribution
            # next_word = torch.multinomial(F.softmax(filtered_logit, dim=-1), 1).squeeze()
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
        halting_probability = torch.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S]
        remainders = torch.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S]
        n_updates = torch.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S, HDD]
        previous_state = torch.zeros_like(inputs).cuda()

        step = 0
        # for l in range(self.num_layers):
        while( ((halting_probability<self.threshold) & (n_updates < max_hop)).byte().any()):
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

            if(decoding):
                state, _, attention_weight = fn((state,encoder_output,[]))
            else:
                # apply transformation on the state
                state = fn(state)

            # update running part in the weighted state and keep the rest
            previous_state = ((state * update_weights.unsqueeze(-1)) + (previous_state * (1 - update_weights.unsqueeze(-1))))
            if(decoding):
                if(step==0):  previous_att_weight = torch.zeros_like(attention_weight).cuda()      ## [B, S, src_size]
                previous_att_weight = ((attention_weight * update_weights.unsqueeze(-1)) + (previous_att_weight * (1 - update_weights.unsqueeze(-1))))
            ## previous_state is actually the new_state at end of hte loop
            ## to save a line I assigned to previous_state so in the next
            ## iteration is correct. Notice that indeed we return previous_state
            step+=1

        if(decoding):
            return previous_state, previous_att_weight, (remainders,n_updates)
        else:
            return previous_state, (remainders,n_updates)

