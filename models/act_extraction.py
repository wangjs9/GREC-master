import os
import json
import torch
import torch.nn as nn
from torch.autograd import Variable

from tqdm import tqdm
from itertools import product
import random
import numpy as np
import copy
from collections import defaultdict
import math
from multiprocessing import Pool
import nltk
stop_word = nltk.corpus.stopwords.words('english')
stop_word += ["gone", "did", "going", "would", "could", "get", "in", "up", "may", "uk", "us", "take", "make", "object", "person", "people"]
import config
from models.common_layer import share_embedding
from dataprocess.data_reader import Lang
import pickle

#TransE
def frequent_triple():
    postC, replyC, dialog_act = data_read()

    triple_list = []
    triple_counter = defaultdict(int)

    forward_act_statistic = defaultdict(lambda: defaultdict(list))
    backword_act_statistic = defaultdict(lambda: defaultdict(list))


    for idx, heads in tqdm(enumerate(postC), total=len(postC)):
        pair = product(heads, replyC[idx])
        for head, tail in pair:
            if head == tail:
                continue
            triple_counter[str([head, dialog_act[idx], tail])] += 1
            forward_act_statistic[tail]['{}'.format(dialog_act[idx])] = head
            backword_act_statistic[head]['{}'.format(dialog_act[idx])] = tail
            if triple_counter[str([head, dialog_act[idx], tail])] == 50:
                triple_list.append([head, dialog_act[idx], tail])
    print(len([x for x in triple_counter.values() if x>=50]))
    print(len(triple_counter))
    np.save(config.data_dict + '/TransE/frequent_triple.npy', triple_list)
    # json.dump(forward_act_statistic, open(config.data_dict + '/TransE/forward_act_statistic.json', 'w'))
    # json.dump(backword_act_statistic, open(config.data_dict + '/TransE/backword_act_statistic.json', 'w'))


class TransE(nn.Module):
    def __init__(self, vocab, act_num, lr=1e-3, weight_decay=1e-5, margin=1.0, C=1.0, device='cuda'):
        super(TransE, self).__init__()
        self.device = device
        self.embed_size = config.emb_dim
        self.begin = Variable(torch.randn([1, 1, self.embed_size]), requires_grad=True).to(self.device)
        self.end = Variable(torch.randn([1, 1, self.embed_size]), requires_grad=True).to(self.device)
        self.act_num = act_num
        self.vocab = vocab
        self.concept_embed = share_embedding(self.vocab, config.pretrain_emb)
        self.act_embed = nn.Embedding(act_num, self.embed_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_F = nn.MarginRankingLoss(margin, reduction='mean').to(device)
        self.C = C

    def scale_loss(self, embedding):
        return torch.sum(
            torch.max(torch.sum(embedding ** 2, dim=1, keepdim=True)-Variable(torch.FloatTensor([1.0])).to(self.device),
                      Variable(torch.FloatTensor([0.0])).to(self.device))
        )

    def distance(self, head, act, tail, test=False):
        distance = head + act - tail
        score = torch.norm(distance, p=2, dim=1)
        if test:
            return score.cpu().detach().numpy()
        else:
            return score

    def load(self, save_path):
        state = torch.load(save_path)
        self.concept_embed = state['concept_embed']
        self.act_embed = state['act_embed']

    def test_one_batch(self, data):
        batch_size = data.size(0)
        head, act, tail = data.unsqueeze(-1).transpose(0, 1)
        head = self.concept_embed(head) + self.begin.repeat(batch_size, 1, 1)
        tail = self.concept_embed(tail) + self.end.repeat(batch_size, 1, 1)
        act = self.act_embed(act)
        score = self.distance(head, act, tail)
        return torch.mean(score, dim=-1)

    def train_one_batch(self, positive, negative):
        batch_size = positive.size(0)
        Phead, Pact, Ptail = positive.unsqueeze(-1).transpose(0, 1)
        Nhead, Nact, Ntail = negative.unsqueeze(-1).transpose(0, 1)

        Phead = self.concept_embed(Phead) + self.begin.repeat(batch_size, 1, 1)
        Ptail = self.concept_embed(Ptail) + self.end.repeat(batch_size, 1, 1)
        Pact = self.act_embed(Pact)

        Nhead = self.concept_embed(Nhead) + self.begin.repeat(batch_size, 1, 1)
        Ntail = self.concept_embed(Ntail) + self.end.repeat(batch_size, 1, 1)
        Nact = self.act_embed(Nact)

        Pos = self.distance(Phead, Pact, Ptail)
        Neg = self.distance(Nhead, Nact, Ntail)
        # print(torch.mean(Pos), torch.mean(Neg))

        concept_embed = torch.cat([Phead, Ptail, Nhead, Ntail]).to(self.device)
        act_embed = torch.cat([Pact, Nact]).to(self.device)

        concept_scale_loss = self.scale_loss(concept_embed)
        act_scale_loss = self.scale_loss(act_embed)

        y = Variable(torch.Tensor([-1])).to(self.device)
        loss = self.loss_F(Pos, Neg, y)

        loss = loss + self.C * (concept_scale_loss/len(concept_embed) + act_scale_loss/len(act_embed))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def save(self, path):
        state = {
            'begin': self.begin,
            'end': self.end,
            'concept_embed': self.concept_embed,
            'act_embed': self.act_embed,
        }
        torch.save(state, path)


class act_info(object):
    def __init__(self, acts, dict=False):
        self.act_num = len(acts)
        if dict:
            self.acts = acts
        else:
            self.acts = {a: index for index, a in enumerate(acts)}

    def act2index(self, a):
        return self.acts[a]

    def acts2list(self):
        return self.acts.keys()


def func(passin):
    dialog, concepts = passin
    postC, replyC, dialog_act = [], [], []
    for key, element in tqdm(dialog.items(), total=len(dialog)):
        conversation = element['conversation']
        act = element['act']

        for i in range(int(len(conversation) / 2)):
            turn = set(conversation[2*i].split())
            postC.append([c for c in list(turn) if c in concepts and c not in stop_word])
            turn = set(conversation[i*2+1].split())
            replyC.append([c for c in list(turn) if c in concepts and c not in stop_word])
            dialog_act.append(act[i*2+1])
    return postC, replyC, dialog_act


def data_read():
    def DivJobList(dialog, concepts, corenum):
        joblist = []
        length = len(dialog)
        unit = math.ceil(length / corenum)

        count, temp = 0, dict()
        for key, ele in dialog.items():
            temp[key] = ele
            count += 1
            if count > unit:
                joblist.append((temp, concepts))
                count, temp = 0, dict()
        joblist.append((temp, concepts))
        return joblist

    if os.path.exists(config.data_dict + '/TransE/TransE_training_data.json'):
        data = json.load(open(config.data_dict + '/TransE/TransE_training_data.json'))
        post_concept, reply_concept, act_relation = data['postC'], data['replyC'], data['dialog_act']
        return post_concept, reply_concept, act_relation

    elif os.path.exists(config.data_dict + '/TransE/format_data_fold_1.json'):
        postC, replyC, dialog_act = [], [], []
        for i in range(6):
            data = json.load(open(config.data_dict + '/TransE/format_data_fold_{}.json'.format(i+1), 'r'))
            pC = data['postC']
            rC = data['replyC']
            act = data['dialog_act']
            postC.extend(pC)
            replyC.extend(rC)
            dialog_act.extend(act)

        post_concept, reply_concept, act_relation = [], [], []
        for idx, pc in tqdm(enumerate(postC), total=len(postC)):
            rc = replyC[idx]
            pc = [w for w in set(pc) if w not in stop_word]
            rc = [w for w in set(rc) if w not in stop_word]
            if pc == [] or rc == []:
                continue
            act = dialog_act[idx]
            post_concept.append(pc)
            reply_concept.append(rc)
            act_relation.append(act)

        data = {'postC': post_concept, 'replyC': reply_concept, 'dialog_act': act_relation}
        json.dump(data, open(config.data_dict + '/TransE/TransE_training_data.json', 'w'))
        return post_concept, reply_concept, act_relation

    else:
        concepts_path = '../conceptnet/concept.txt'
        empathetic_dialog_path = '../data/empathetic_dialogues/train/train(act-cause).json'

        with open(concepts_path, 'r') as f:
            concepts = f.readlines()
            concepts = [c.strip() for c in concepts]

        empathetic_dialog = json.load(open(empathetic_dialog_path, 'r'))
        corenum = 5
        pool = Pool(corenum)
        joblist = DivJobList(empathetic_dialog, concepts, corenum)
        return_dict = pool.map(func, joblist)
        pool.close()
        pool.join()

        postC, replyC, dialog_act = [], [], []
        for idx in range(corenum):
            pC, rC, act = return_dict[idx]
            postC.extend(pC)
            replyC.extend(rC)
            dialog_act.extend(act)
            data = {'postC': pC, 'replyC': rC, 'dialog_act': act}
            json.dump(data, open(config.data_dict + '/TransE/format_data_fold_{}.json'.format(idx+1), 'w'))


        post_concept, reply_concept, act_relation = [], [], []
        for idx, pc in tqdm(enumerate(postC), total=len(postC)):
            rc = replyC[idx]
            pc = [w for w in set(pc) if w not in stop_word]
            rc = [w for w in set(rc) if w not in stop_word]
            if pc == [] or rc == []:
                continue
            act = dialog_act[idx]
            post_concept.append(pc)
            reply_concept.append(rc)
            act_relation.append(act)

        data = {'postC': post_concept, 'replyC': reply_concept, 'dialog_act': act_relation}
        json.dump(data, open(config.data_dict + '/TransE/TransE_training_data.json', 'w'))

        return post_concept, reply_concept, act_relation



def process_data(postC, replyC, dialog_act):
    if not os.path.exists('./data/empathetic_dialogues/vocab.pkl'):
        vocab = Lang(
        {idx: word for idx, word in enumerate(np.load('./data/empathetic_dialogues/vocab.npy', allow_pickle=True))})
        pickle.dump(vocab, open('./data/empathetic_dialogues/vocab.pkl', 'wb'))
    else:
        vocab = pickle.load(open('./data/empathetic_dialogues/vocab.pkl', 'rb'))

    ACT = act_info(list(set(dialog_act)))
    json.dump(ACT.acts, open(config.data_dict + '/TransE/acts_info.json', 'w'))
    if os.path.exists(config.data_dict + '/TransE/training_triple.json'):
        data = json.load(open(config.data_dict + '/TransE/training_triple.json', 'r'))
        triple_list = data['positive']
        negative = data['negative']
    else:
        act_statistic = {act: defaultdict(int) for act in set(dialog_act)}
        triple_list = []
        entities = list(vocab.word2index.values())
        head_num = defaultdict(list)
        tail_num = defaultdict(list)
        assert len(postC) == len(replyC) == len(dialog_act)
        triple_counter = defaultdict(int)
        for idx, heads in tqdm(enumerate(postC), total=len(postC)):
            pair = product(heads, replyC[idx])
            for head, tail in pair:
                if head == tail:
                    continue
                try:
                    h = vocab.word2index[head]
                    t = vocab.word2index[tail]
                except KeyError:
                    print(head, tail)
                a = ACT.act2index(dialog_act[idx])

                triple_counter[str([h, a, t])] += 1

                if triple_counter[str([h, a, t])] == 10:
                    triple_list.append([h, a, t])
                    head_num[h].append(t)
                    tail_num[t].append(h)
                    act_statistic[dialog_act[idx]][tail] += 1

        json.dump(act_statistic, open(config.data_dict + '/TransE/act_statistic.json', 'w'))
        head_num = {key: len(set(ele)) for key, ele in head_num.items()}
        tail_num = {key: len(set(ele)) for key, ele in tail_num.items()}

        random.shuffle(triple_list)

        negative = []
        for triple in tqdm(triple_list, total=len(triple_list)):
            neg_triple = copy.deepcopy(triple)
            h, a, t = neg_triple
            pr = np.random.random(1)[0]
            p = head_num[h] / (head_num[h]+tail_num[t])

            if pr < p:
                neg_triple[0] = random.sample(entities, 1)[0]
                while neg_triple in triple_list:
                    neg_triple[0] = random.sample(entities, 1)[0]
            else:
                neg_triple[-1] = random.sample(entities, 1)[0]
                while neg_triple in triple_list:
                    neg_triple[-1] = random.sample(entities, 1)[0]

            negative.append(neg_triple)

        data = {'positive': triple_list, 'negative': negative}

        json.dump(data, open(config.data_dict + '/TransE/training_triple.json', 'w'))

    triple_list = torch.Tensor(np.array(triple_list)).long().cuda()
    negative = torch.Tensor(np.array(negative)).long().cuda()

    valid_len = int(len(triple_list) * 0.2) if len(triple_list) < 5000 else 1000
    valid_triple = triple_list[:valid_len]
    train_triple = triple_list[valid_len:]

    valid_negative_triple = negative[:valid_len]
    train_negative_triple = negative[valid_len:]

    train_data = (train_triple, train_negative_triple)
    valid_data = (valid_triple, valid_negative_triple)

    return train_data, valid_data, ACT, vocab


class Dataloader(object):
    def __init__(self, data, batch_size, device='cuda', test=False):
        self.test = test
        if test:
            self.postive = data.to(device)
        else:
            self.postive = data[0].to(device)
            self.negative = data[1].to(device)
        self.hard_length = len(self.postive)
        self.length = math.ceil(self.hard_length/batch_size)
        self.batch_size = batch_size
        self.flag = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.flag < self.hard_length:
            start = copy.deepcopy(self.flag)
            end = self.flag + self.batch_size
            if end > self.hard_length:
                end = self.hard_length
            self.flag = end
            if self.test:
                return self.postive[start:end]
            else:
                return (self.postive[start:end], self.negative[start:end])
        else:
            raise StopIteration

    def __len__(self):
        return self.length


def train_TransE():
    batch_size, epochs = 1024, 200
    lr, weight_lr = 1e-2, 1e-4
    postC, replyC, dialog_act = data_read()
    trainset, validset, ACT, vocab = process_data(postC, replyC, dialog_act)

    model = TransE(vocab, ACT.act_num, lr, weight_lr)
    model.to('cuda')

    epoch = 0
    while True:
        model.train()
        epoch += 1
        train_data = Dataloader(trainset, batch_size)
        for positive, negative in iter(train_data):
            loss = model.train_one_batch(positive, negative)

        if epoch > 600:
            valid_loss = 0
            model.eval()
            valid_data = Dataloader(validset, batch_size)
            for pos, neg in iter(valid_data):
                valid_loss += model.train_one_batch(pos, neg)
            print('Epoch {}: training loss: {}, testing loss {}'.format(epoch, loss, valid_loss))
            if valid_loss < 20.25:
                model.save(config.data_dict + '/TransE/model.mdl')
                break


def augment_net(testing=True):
    acts_info = json.load(open(config.data_dict + '/TransE/acts_info.json', 'r'))
    ACT = act_info(acts_info, dict=True)
    vocab = pickle.load(open('./data/empathetic_dialogues/vocab.pkl', 'rb'))
    model = TransE(vocab, ACT.act_num)
    model.load(config.data_dict + '/TransE/model.mdl')
    data = json.load(open(config.data_dict + '/TransE/training_triple.json', 'r'))
    triple_list = data['positive']
    if testing:
        # ground truth testing
        negative = data['negative']

        positive = torch.Tensor(np.array(triple_list)).long().cuda()
        negative = torch.Tensor(np.array(negative)).long().cuda()
        model.cuda()
        model.eval()

        average_pos = model.test_one_batch(positive)
        average_neg = model.test_one_batch(negative)

        print(torch.mean(average_pos).item(), torch.mean(average_neg).item())
        print(torch.topk(average_pos, int(0.01*len(average_pos)), largest=True, sorted=True)[0])
        print(torch.topk(average_neg, int(0.01*len(average_neg)), largest=False, sorted=True)[0])
    # 0.50; 1.17



if __name__ == '__main__':
    # train_TransE()
    # augment_net()
    # frequent_triple()
    pass