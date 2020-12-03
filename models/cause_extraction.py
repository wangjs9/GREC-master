import warnings
warnings.filterwarnings('ignore')

import torch, os, time, random, pickle
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import BertModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models.common_layer import RTHNLayer, _gen_bias_mask
import configparser
config = configparser.ConfigParser()
config.read("./utils/params.cfg")

emotion_dict = {'NA': 0, 'SADNESS': 1, 'ANGER': 2, 'FEAR': 3, 'SURPRISE': 4,
                'ANTICIPATION': 5, 'TRUST': 6, 'DISGUST': 7, 'OTHER': 8, 'JOY': 0}

class DatasetIterator(object):
    def __init__(self, batch_size, dataset, device):

        self.batch_size = batch_size
        self.dataset = dataset

        self.total_batch = len(self.dataset)
        self.n_batches = self.total_batch // self.batch_size
        self.residue = False if self.n_batches * self.batch_size == self.total_batch else True
        self.start = 0
        self.end = self.start + self.batch_size
        self.device = device

    def _to_tensor(self, data):
        x = torch.LongTensor([_[0] for _ in data]).to(self.device)
        mask = torch.LongTensor([_[1] for _ in data]).to(self.device)
        y = torch.LongTensor([_[2] for _ in data]).to(self.device)
        sen_len = torch.LongTensor([_[3] for _ in data]).to(self.device)
        doc_len = torch.LongTensor([_[4] for _ in data]).to(self.device)
        relative_pos = torch.LongTensor([_[5] for _ in data]).to(self.device)
        emotion = torch.LongTensor([_[6] for _ in data]).to(self.device)

        return x, mask, y, sen_len, doc_len, relative_pos, emotion

    def __next__(self):
        if self.end > self.total_batch:
            batches = self.dataset[self.start: ] + self.dataset[:self.end % self.total_batch]
        else:
            batches = self.dataset[self.start: self.end]
        self.start = self.end % self.total_batch
        self.end = self.start + self.batch_size

        batches = self._to_tensor(batches)
        return batches

    def __iter__(self):
        return self

class PredDataIterator(object):
    def __init__(self, batch_size, dataset, device):

        self.batch_size = batch_size
        self.dataset = dataset

        self.total_batch = len(self.dataset)
        self.n_batches = self.total_batch // self.batch_size
        self.residue = False if self.n_batches * self.batch_size == self.total_batch else True
        self.start = 0
        self.end = (self.start + self.batch_size) if (self.start + self.batch_size) < self.total_batch else self.total_batch
        self.device = device

    def _to_tensor(self, data):
        x = torch.LongTensor([_[0] for _ in data]).to(self.device)
        mask = torch.LongTensor([_[1] for _ in data]).to(self.device)
        sen_len = torch.LongTensor([_[2] for _ in data]).to(self.device)
        doc_len = torch.LongTensor([_[3] for _ in data]).to(self.device)
        relative_pos = torch.LongTensor([_[4] for _ in data]).to(self.device)
        emotion = torch.LongTensor([_[5] for _ in data]).to(self.device)

        return x, mask, sen_len, doc_len, relative_pos, emotion

    def __next__(self):
        if self.start == self.total_batch:
            raise StopIteration('Cannot get the next batch.')
        batches = self.dataset[self.start: self.end]
        self.start = self.end
        self.end = (self.start + self.batch_size) if (self.start + self.batch_size) < self.total_batch else self.total_batch
        batches = self._to_tensor(batches)
        return batches

    def __iter__(self):
        return self

def DataIter(data_path, batch_size, device):

    def load_data(data_path):
        x = pickle.load(open(data_path + 'x.txt', 'rb'))
        mask = pickle.load(open(data_path + 'mask.txt', 'rb'))
        y = pickle.load(open(data_path + 'y.txt', 'rb'))
        sen_len = pickle.load(open(data_path + 'sen_len.txt', 'rb'))
        doc_len = pickle.load(open(data_path + 'doc_len.txt', 'rb'))
        relative_pos = pickle.load(open(data_path + 'relative_pos.txt', 'rb'))
        emotion = pickle.load(open(data_path + 'emotion.txt', 'rb'))
        print('x.shape {}\nmask.shape {}\ny.shape {} \nsen_len.shape {} \ndoc_len.shape '
              '{}\nrelative_pos.shape {}\nemotion.shape {}'.format(x.shape, mask.shape,
                y.shape, sen_len.shape, doc_len.shape, relative_pos.shape, emotion.shape))

        dataset = list()

        for i in range(len(x)):
            dataset.append([x[i], mask[i], y[i], sen_len[i], doc_len[i], relative_pos[i], emotion[i]])
        random.shuffle(dataset)
        return dataset

    dataset = load_data(data_path)
    total_len = len(dataset)
    train_len = int(total_len * 0.9)
    train_data = dataset[:train_len]
    dev_data = dataset[train_len:total_len]

    train_DataIter = DatasetIterator(batch_size, train_data, device)
    dev_DataIter = DatasetIterator(batch_size, dev_data, device)

    return train_DataIter, dev_DataIter

class RTHN(nn.Module):
    def __init__(self, config=config['ece'], num_heads=8, use_mask=False, input_dropout=0.0,
                 word_dropout=0.0, layer_dropout=0.0, attention_dropout=0.0, lr=1e-3, l2_reg=1e-5):
        super(RTHN, self).__init__()

        self.model_dir = config['model_dir'] + '/layer-{}/'.format(config['n_layers'])
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.max_seq_len = int(config['max_seq_len'])
        self.max_doc_len = int(config['max_doc_len'])

        ## word embedding
        self.embed_dim = 768
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=False,
                    output_hidden_states=False)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.input_dropout = nn.Dropout(input_dropout)

        ## position embedding
        self.embed_dim_pos = 56
        pos_embedding = torch.FloatTensor(np.load(config['pos_embedding_path'], allow_pickle=True))
        self.pos_embed = nn.Embedding.from_pretrained(pos_embedding, freeze=True)

        ## word level encoding
        self.hidden_size = int(config['hidden_size'])
        self.WordEncoder = nn.LSTM(self.embed_dim, self.hidden_size, bidirectional=True, batch_first=True, dropout=word_dropout)

        self.wordlinear_1 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        self.wordlinear_2 = nn.Linear(self.hidden_size * 2, 1, bias=False)
        self.classlinear = nn.Linear(self.hidden_size * 2, 1)

        ## sentence level encoding
        self.n_layers = int(config['n_layers'])
        self.program_class = 2

        params_layer_1 = ((self.hidden_size * 2 + self.embed_dim_pos,
                          self.hidden_size * 2 + self.embed_dim_pos,
                          self.hidden_size * 2),
                          self.hidden_size * 2 + self.embed_dim_pos,
                          self.hidden_size * 2 + self.embed_dim_pos,
                          num_heads,
                          self.hidden_size * 2,
                          self.program_class,
                          self.max_doc_len,
                          _gen_bias_mask(config['max_doc_len']) if use_mask else None,
                          layer_dropout,
                          attention_dropout)

        params_layers = ((self.hidden_size * 2 + self.max_doc_len,
                         self.hidden_size * 2 + self.max_doc_len,
                         self.hidden_size * 2),
                         self.hidden_size * 2 + self.max_doc_len,
                         self.hidden_size * 2 + self.max_doc_len,
                         num_heads,
                         self.hidden_size * 2,
                         self.program_class,
                         self.max_doc_len,
                         _gen_bias_mask(self.max_doc_len) if use_mask else None,
                         layer_dropout,
                         attention_dropout)
        self.rthn = nn.ModuleList([RTHNLayer(*params_layer_1)]+[RTHNLayer(*params_layers) for _ in range(self.n_layers)])
        ## training
        self.l2_reg = l2_reg

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        ## model save path
        self.best_path = config['model_dir'] + '/layer-{}/model'.format(config['n_layers'], self.n_layers)
        if os.path.exists(self.best_path):
            print("loading weights")
            state = torch.load(self.best_path, map_location=lambda storage, location: storage)
            self.bert.load_state_dict(state['bert_state_dict'])
            self.WordEncoder.load_state_dict(state['WordEncoder_state_dict'])
            self.wordlinear_1.load_state_dict(state['wordlinear_1_state_dict'])
            self.wordlinear_2.load_state_dict(state['wordlinear_2_state_dict'])
            self.classlinear.load_state_dict(state['classlinear_state_dict'])
            self.rthn.load_state_dict(state['rthn_state_dict'])

            if (config['load_optim']):
                self.optimizer.load_state_dict(state['optimizer_state_dict'])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

            # self.eval()

    def forward(self, input_ids, attention_mask, output_ids, sen_len, doc_len, word_dis, emotion, train=True):
        self.IsTrain = train
        self.device = input_ids.device

        # bitpos = output_ids[:, :, 1] == 1
        # bitmask = (torch.FloatTensor(bitpos.size()).uniform_().to(self.device) > 0.3).long() * bitpos.long()
        # bitmask = bitmask.unsqueeze(-1) * torch.LongTensor([0, -1]).reshape(1, 1, 2).to(self.device)
        # mask_output_ids = output_ids + bitmask

        ### word level encoder based on RNNs
        clause, _ = self.bert(input_ids.reshape(-1, self.max_seq_len), token_type_ids=None, attention_mask=attention_mask.reshape(-1, self.max_seq_len))
        clause = self.input_dropout(clause).reshape(-1, self.max_seq_len, self.embed_dim)
        word_encode, _ = self.WordEncoder(clause)
        word_encode = word_encode.view(-1, self.max_seq_len, 2, self.hidden_size)
        word_encode = torch.cat((word_encode[:,:,-2], word_encode[:,:,-1]), dim=-1) # (batch_size * doc_len, seq_len, hidden_size *2)
            ### attention of word level encoding
        alpha = self.wordlinear_2(self.wordlinear_1(word_encode)).reshape(-1, 1, self.max_seq_len) # (batch_size * doc_len, 1, seq_len)
        mask = torch.arange(0, self.max_seq_len, step=1).to(self.device).repeat(alpha.size(0), 1) \
               < sen_len.reshape(-1, 1)
        alpha = torch.exp(alpha) * mask.unsqueeze(1).float()
        alpha = torch.softmax(alpha, dim=-1)
        sen_encode = torch.matmul(alpha, word_encode).reshape(-1, self.max_doc_len, self.hidden_size * 2) # (batch_size, doc_len, hidden_size *2)
        # sen_encode = torch.relu(sen_encode)
            ### end of attention
        ### end of word level encoding

        ### clause level encoder based on Transformer
        attn_mask = torch.arange(0, self.max_doc_len, step=1).to(self.device).expand(sen_encode.size()[:2]) \
                    < doc_len.reshape(-1, 1)

        word_dis = self.pos_embed(word_dis)  # (batch_size, doc_len, seq_len, embed_size)
        word_dis = word_dis[:, :, 0, :].reshape(-1, self.max_doc_len,
                                                self.embed_dim_pos)  # (batch_size, doc_len, pos_embed_size)
        sen_encode_value = torch.cat((sen_encode, word_dis),
                                     dim=-1)  # (batch_size, doc_len, hidden_size * 2 + pos_embed_size)
        for l in range(self.n_layers+1):
            sen_encode, pred, pred_assist_label, reg = self.rthn[l](sen_encode_value, sen_encode, attn_mask)
            # shape --> sen_encode: (batch_size, doc_len, hidden_size * 2)
            # shape --> pred_label: (batch_size, doc_len, doc_len)
            if l == 0:
                pred_assist_label_list = torch.empty_like(pred_assist_label.unsqueeze(0))
            pred_assist_label_list = torch.cat((pred_assist_label_list, pred_assist_label.unsqueeze(0)), dim=0)
            if l > 0:
                pred_assist_label = torch.div(torch.sum(pred_assist_label_list, dim=0), l)
            sen_encode_value = torch.cat((sen_encode, pred_assist_label), dim=-1)

        pred = pred * attn_mask.float().unsqueeze(-1) + torch.ones_like(pred) * (~attn_mask).float().unsqueeze(-1) * 1e-18

        valid_num = torch.sum(doc_len).to(self.device)
        loss_weight = output_ids * torch.FloatTensor([10, 1]).to(self.device)
        loss = - torch.sum(loss_weight * torch.log(pred)) / valid_num + reg * self.l2_reg

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        true_y_op = torch.argmax(output_ids, -1).cpu().numpy()
        pred_y_op = torch.argmax(pred, -1).cpu().numpy()

        accuracy, precision, recall, F1 = self.compute_score(true_y_op, pred_y_op, doc_len)

        return accuracy, precision, recall, F1

    def save_model(self, iter, accuracy, precision, recall, F1):

        state = {
            'iter': iter,
            'bert_state_dict': self.bert.state_dict(),
            'WordEncoder_state_dict': self.WordEncoder.state_dict(),
            'wordlinear_1_state_dict': self.wordlinear_1.state_dict(),
            'wordlinear_2_state_dict': self.wordlinear_2.state_dict(),
            'classlinear_state_dict': self.classlinear.state_dict(),
            'rthn_state_dict': self.rthn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        model_save_path = os.path.join(self.model_dir,
                                       'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(iter, accuracy, precision, recall, F1))

        torch.save(state, model_save_path)

    def predict(self, input_ids, attention_mask, sen_len, doc_len, word_dis, emotion):
        self.device = input_ids.device

        ### word level encoder based on RNNs
        clause, _ = self.bert(input_ids.reshape(-1, self.max_seq_len), token_type_ids=None,
                              attention_mask=attention_mask.reshape(-1, self.max_seq_len))
        clause = self.input_dropout(clause).reshape(-1, self.max_seq_len, self.embed_dim)
        word_encode, _ = self.WordEncoder(clause)
        word_encode = word_encode.view(-1, self.max_seq_len, 2, self.hidden_size)
        word_encode = torch.cat((word_encode[:, :, -2], word_encode[:, :, -1]),
                                dim=-1)  # (batch_size * doc_len, seq_len, hidden_size *2)
        ### attention of word level encoding
        alpha = self.wordlinear_2(self.wordlinear_1(word_encode)).reshape(-1, 1,
                                                                          self.max_seq_len)  # (batch_size * doc_len, 1, seq_len)
        mask = torch.arange(0, self.max_seq_len, step=1).to(self.device).repeat(alpha.size(0), 1) \
               < sen_len.reshape(-1, 1)
        alpha = torch.exp(alpha) * mask.unsqueeze(1).float()
        alpha = torch.softmax(alpha, dim=-1)
        sen_encode = torch.matmul(alpha, word_encode).reshape(-1, self.max_doc_len,
                                                              self.hidden_size * 2)  # (batch_size, doc_len, hidden_size *2)
        ### end of attention
        ### end of word level encoding

        ### clause level encoder based on Transformer
        attn_mask = torch.arange(0, self.max_doc_len, step=1).to(self.device).expand(sen_encode.size()[:2]) \
                    < doc_len.reshape(-1, 1)

        word_dis = self.pos_embed(word_dis)  # (batch_size, doc_len, seq_len, embed_size)
        word_dis = word_dis[:, :, 0, :].reshape(-1, self.max_doc_len,
                                                self.embed_dim_pos)  # (batch_size, doc_len, pos_embed_size)
        sen_encode_value = torch.cat((sen_encode, word_dis),
                                     dim=-1)  # (batch_size, doc_len, hidden_size * 2 + pos_embed_size)
        for l in range(self.n_layers + 1):
            sen_encode, pred, pred_assist_label, reg = self.rthn[l](sen_encode_value, sen_encode, attn_mask)
            # shape --> sen_encode: (batch_size, doc_len, hidden_size * 2)
            # shape --> pred_label: (batch_size, doc_len, doc_len)
            if l == 0:
                pred_assist_label_list = torch.empty_like(pred_assist_label.unsqueeze(0))
            pred_assist_label_list = torch.cat((pred_assist_label_list, pred_assist_label.unsqueeze(0)), dim=0)
            if l > 0:
                pred_assist_label = torch.div(torch.sum(pred_assist_label_list, dim=0), l)
            sen_encode_value = torch.cat((sen_encode, pred_assist_label), dim=-1)

        pred = pred * attn_mask.float().unsqueeze(-1) + torch.ones_like(pred) * (~attn_mask).float().unsqueeze(-1) * 1e-18

        pred_y_op = torch.argmax(pred, -1).cpu().numpy()
        y_pred = list()
        for i in range(len(pred_y_op)):
            y_pred.append(pred_y_op[i][:doc_len[i]-1].tolist())

        return y_pred

    def compute_score(self, pred_y, true_y, doc_len):
        tmp1, tmp2 = [], []
        for i in range(len(pred_y)):
            for j in range(doc_len[i]):
                tmp1.append(pred_y[i][j])
                tmp2.append(true_y[i][j])

        y_pred, y_true = np.array(tmp1), np.array(tmp2)
        if not self.IsTrain:
            print(y_pred)
            print(y_true)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', pos_label=0)
        recall = recall_score(y_true, y_pred, average='binary', pos_label=0)
        F1 = f1_score(y_true, y_pred, average='binary', pos_label=0)
        return accuracy, precision, recall, F1

def train():
    localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("***********localtime: ", localtime)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    train_DataIter, dev_DataIter = DataIter(config['ece']['training_dir'], 64, device)
    model = RTHN()
    model.to(device)

    try:
        model = model.train()
        check_iter, best_F1, best_acc, patient = 50, 0, 0, 0

        for n_iter in tqdm(range(20000)):
            accuracy_train, precision_train, recall_train, F1_train = model(*next(train_DataIter))

            if (n_iter + 1) % check_iter == 0:
                model = model.eval()
                accuracy, precision, recall, F1 = model(*next(train_DataIter), train=False)
                model = model.train()

                if F1 >= best_F1 and accuracy >= best_acc:
                    if check_iter > 300:
                        check_iter = 200
                    best_F1 = F1
                    best_acc = accuracy
                    patient = 0
                    model.save_model(n_iter, accuracy, precision, recall, F1)

                elif n_iter > 900:
                    patient += 1

                if patient > 10:
                    break
                print("TRAIN: accuracy:{:.2f} precision:{:.2f} recall:{:.2f} F1:{:.2f}".format(accuracy_train, precision_train, recall_train, F1_train))
                print("EVAL: accuracy:{:.2f} precision:{:.2f} recall:{:.2f} F1:{:.2f}".format(accuracy, precision, recall, F1))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    print('Finish training model')

def predict(situation, history, targets, DataType, datapath):
    if DataType not in ['train', 'valid', 'test']:
        raise ValueError('`DataType` must be in [`train`, `valid`, `test`]')
    localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("***********localtime: ", localtime)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    UserIter, BotIter, userIdx, botIdx, causetext = load_test_data(situation, history, targets, device)
    model = RTHN()
    model.to(device)
    model = model.eval()
    user_res, bot_res = list(), list()

    while True:
        try:
            res = model.predict(*(next(UserIter)))
            user_res.extend(res)
        except StopIteration:
            break

    while True:
        try:
            res = model.predict(*(next(BotIter)))
            bot_res.extend(res)
        except StopIteration:
            break

    usercause, botcause, usercause_labels, botcause_labels = [], [], [], []

    tmp, tmp_labels = [], []
    curId = 0
    for idx, convId in enumerate(userIdx):
        if curId != convId:
            usercause.append(tmp.copy())
            usercause_labels.append(tmp_labels.copy())
            tmp, tmp_labels = [], []
            curId = convId
        predres = user_res[idx]
        clauseset = causetext[convId]
        clause = ''
        for j, pred in enumerate(predres):
            if pred == 0:
                clause = ' ' + clauseset[j]
        tmp_labels.append(predres)
        tmp.append(clause)
    usercause.append(tmp.copy())
    usercause_labels.append(tmp_labels.copy())

    tmp, tmp_labels = [], []
    curId = 0
    for idx, convId in enumerate(botIdx):
        if curId != convId:
            botcause.append(tmp.copy())
            botcause_labels.append(tmp_labels.copy())
            tmp, tmp_labels = [], []
            curId = convId
        predres = bot_res[idx]
        clauseset = causetext[convId]
        clause = ''
        for j, pred in enumerate(predres):
            if pred == 0:
                clause = ' ' + clauseset[j]
        tmp_labels.append(predres)
        tmp.append(clause)
    botcause.append(tmp.copy())
    botcause_labels.append(tmp_labels.copy())

    print('No of instances:', len(usercause), len(botcause), len(usercause_labels), len(botcause_labels))

    if DataType == 'train':
        num = 200
    else:
        num = 20

    np.save(datapath + 'sys_usercause_labels.{}.npy'.format(DataType), usercause_labels)
    np.save(datapath + 'min_usercause_labels.{}.npy'.format(DataType), usercause_labels[:num])

    np.save(datapath + 'sys_botcause_labels.{}.npy'.format(DataType), botcause_labels)
    np.save(datapath + 'min_botcause_labels.{}.npy'.format(DataType), botcause_labels[:num])

    np.save(datapath + 'sys_usercause_texts.{}.npy'.format(DataType), usercause)
    np.save(datapath + 'min_usercause_texts.{}.npy'.format(DataType), usercause[:num])

    np.save(datapath + 'sys_botcause_texts.{}.npy'.format(DataType), botcause)
    np.save(datapath + 'min_botcause_texts.{}.npy'.format(DataType), botcause[:num])


def load_test_data(situation, history, targets, device, max_doc_len=32, max_sen_len=50, batch_size=64):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    print('load data...')
    total_len= len(situation)
    userIdx, user_data, botIdx, bot_data = [], [], [], []
    causetext = []
    for idx in range(total_len):
        pos = 0
        x, mask, sen_len, doc_len = [], [], [], []
        causeclause = []
        emotions = np.zeros((max_doc_len,))
        for sen in situation[idx]:
            causeclause.append(sen)
            encoded_dict = tokenizer.encode_plus(sen, add_special_tokens=True,
                                                max_length=max_sen_len, pad_to_max_length=True,
                                                return_attention_mask=True, truncation=True)
            clause = encoded_dict['input_ids']
            x.append(clause)
            attention_mask = encoded_dict['attention_mask']
            mask.append(attention_mask)
            sen_len.append(sum(encoded_dict['attention_mask']))
            pos += 1

        for sentences in history[idx][:-1]:
            for sen in sentences:
                causeclause.append(sen)
                encoded_dict = tokenizer.encode_plus(sen, add_special_tokens=True,
                                                     max_length=max_sen_len, pad_to_max_length=True,
                                                     return_attention_mask=True, truncation=True)
                clause = encoded_dict['input_ids']
                x.append(clause)
                attention_mask = encoded_dict['attention_mask']
                mask.append(attention_mask)
                sen_len.append(sum(encoded_dict['attention_mask']))
                pos += 1

        for sen in history[idx][-1]:
            causeclause.append(sen)
            encoded_dict = tokenizer.encode_plus(sen, add_special_tokens=True,
                                                 max_length=max_sen_len, pad_to_max_length=True,
                                                 return_attention_mask=True, truncation=True)
            clause = encoded_dict['input_ids']
            x.append(clause)
            attention_mask = encoded_dict['attention_mask']
            mask.append(attention_mask)
            sen_len.append(sum(encoded_dict['attention_mask']))
            pos += 1
            tmp_x, tmp_mask, tmp_sen_len = x.copy(), mask.copy(), sen_len.copy()
            tmp_relative_pos = [np.array([69 - pos + i + 1] * max_sen_len) for i in range(pos)]
            for j in range(max_doc_len - pos):
                tmp_x.append(np.zeros((max_sen_len,)))
                tmp_mask.append(np.zeros(max_sen_len,))
                tmp_sen_len.append(0)
                tmp_relative_pos.append(np.zeros((max_sen_len,)))
            tmp = [tmp_x, tmp_mask, tmp_sen_len, pos, tmp_relative_pos, emotions]
            user_data.append(tmp)
            userIdx.append(idx)

        for sen in targets[idx]:
            if pos == 32:
                break
            causeclause.append(sen)
            encoded_dict = tokenizer.encode_plus(sen, add_special_tokens=True,
                                                 max_length=max_sen_len, pad_to_max_length=True,
                                                 return_attention_mask=True, truncation=True)
            clause = encoded_dict['input_ids']

            x.append(clause)
            attention_mask = encoded_dict['attention_mask']
            mask.append(attention_mask)
            sen_len.append(sum(encoded_dict['attention_mask']))
            pos += 1
            tmp_x, tmp_mask, tmp_sen_len = x.copy(), mask.copy(), sen_len.copy()
            tmp_relative_pos = [np.array([69 - pos + i + 1] * max_sen_len) for i in range(pos)]
            for j in range(max_doc_len - pos):
                tmp_x.append(np.zeros((max_sen_len,)))
                tmp_mask.append(np.zeros(max_sen_len,))
                tmp_sen_len.append(0)
                tmp_relative_pos.append(np.zeros((max_sen_len,)))
            tmp = [tmp_x, tmp_mask, tmp_sen_len, pos, tmp_relative_pos, emotions]
            bot_data.append(tmp)
            botIdx.append(idx)

        causetext.append(causeclause.copy())

    UserIter = PredDataIterator(batch_size, user_data, device)
    BotIter = PredDataIterator(batch_size, bot_data, device)

    print(len(situation))
    print('***Total {} instances for queries, and {} instances for responses***'.format(len(user_data), len(bot_data)))

    return UserIter, BotIter, userIdx, botIdx, causetext
