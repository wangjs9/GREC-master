import torch
import torch.utils.data as data
import logging
import numpy as np
import config
import pprint
pp = pprint.PrettyPrinter(indent=1)
from models.common_layer import write_config
from dataprocess.data_reader import load_dataset

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data
        self.emo_map = {
            'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7,
            'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13,
            'anxious': 14, 'disappointed': 15,
            'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21,
            'anticipating': 22, 'embarrassed': 23,
            'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29,
            'apprehensive': 30, 'faithful': 31}

        self.rel_map = {0: ' be opposite to ', 1: ' be a typical location for ', 2: ' can typically do is ',
            3: ' cause ', 4: ' make someone want ', 5: ' be created by the process of ',
            6: ' can be more explanatory through ', 7: ' be a word or phrase that appears within and contributes to ',
            8: ' is a conscious entity that typically wants ', 9: ' be in the same set of and distinct from ',
            10: ' happen at the same time of ', 11: ' be derived from ', 12: ' have a common origin like ',
            13: ' is an inflected form of ', 14: ' have an inherent part or a social construct of possession ',
            15: ' , in a topic area, technical field, or regional dialect, be a word used in the context of ',
            16: ' be an event that begins with subevent ', 17: ' be an event that concludes with subevent ',
            18: ' happen requires the happening of ', 19: ' can be described as or have a property ',
            20: ' have a subevent ', 21: ' be an example of ', 22: ' is a subtype or a specific instance of ',
            23: ' be near ', 24: ' be made of ', 25: ' be a specific way to do ', 26: ' be a step toward accomplishing the goal ',
            27: ' be not the capable of ', 28: ' be a conscious entity that typically wants ', 29: ' can not be described as or have a property ',
            30: ' be part of ', 31: ' can be done through ', 32: ' be conceptually related to ',
            33: ' is similar to ', 34: ' symbolically represents ', 35: ' have very similar meanings to ',
            36: ' be used for ', 37: ' be the capital of ', 38: ' field ', 39: ' genre ', 40: ' genus ', 41: ' be influenced by ',
            42: ' be known for ', 43: ' language ', 44: ' leader ', 45: ' occupation ', 46: ' product '
            }

    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["situation_text"] = self.data["situation"][index]
        item["dialog_text"] = self.data["dialog"][index]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = self.data["emotion"][index]
        item["cause_text"] = self.data["usercause"][index]
        item["botcause_text"] = self.data["botcause"][index]

        item["cause_label"] = self.data["usercause_label"][index]
        item["botcause_label"] = self.data["botcause_label"][index]
        graphIdx = self.data["graphidx"][index]

        item["graphs"] = [self.preprocess_graph(self.data["graph"][id]) for id in graphIdx]

        item["context"], item["context_mask"] = self.preprocess((item["situation_text"], item["dialog_text"]))
        item["cause_batch"] = self.preprocess(["CLS"]+[config.CLS_idx])

        item["context_text"] = item["situation_text"] + [ele for lst in item["dialog_text"] for ele in lst]
        item["clause"] = []
        item["causepos"] = []
        for num, text in enumerate(item["context_text"]):
            clause = self.preprocess(text, clause=True)
            item["clause"].append(clause)
            score = 0
            for i, label in enumerate(len(item["cause_label"])):
                try:
                    score += label[num]
                except IndexError:
                    continue
            for i in range(len(clause)):
                item["causepos"].append(score)
        max_score = max(item["causepos"])
        item["causepos"] = [65+max_score-score if score else 0 for score in item["causepos"]]

        item["target"] = self.preprocess(item["target_text"]+["EOS"])
        item["emotion"], item["emotion_label"] = self.preprocess_emo(item["emotion_text"], self.emo_map)

        return item

    def preprocess(self, arr, clause=False):
        """Converts words to ids."""
        if clause:
            sequence = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                        arr]
            return torch.LongTensor(sequence)

        else:
            situation, context = arr
            X_dial = [config.CLS_idx]
            X_mask = [config.CLS_idx]
            for i, sentence in enumerate(situation):
                X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                           sentence]
                X_mask += [self.vocab.word2index["SIT"] for _ in range(len(sentence))]

            for i, sentences in enumerate(context):
                for j, sentence in enumerate(sentences):
                    X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                               sentence]
                    # >>>>>>>>>> spk: whether this sen is from a user or bot >>>>>>>>>> #
                    spk = self.vocab.word2index["USR"] if i % 2 == 0 else self.vocab.word2index["SYS"]
                    X_mask += [spk for _ in range(len(sentence))]

            assert len(X_dial) == len(X_mask)

            return torch.LongTensor(X_dial), torch.LongTensor(X_mask)
            # >>>>>>>>>> context, context mask >>>>>>>>>> #

    def preprocess_emo(self, emotion, emo_map):
        program = [0] * len(emo_map)
        program[emo_map[emotion]] = 1
        # >>>>>>>>>> one hot mode and label mode >>>>>>>>>> #
        return program, emo_map[emotion]

    def preprocess_graph(self, graph):
        concepts = graph["concepts"]
        heads = graph["head_ids"]
        tails = graph["tail_ids"]
        relations = graph["relations"]
        _relations = []
        for idx, rel in enumerate(relations):
            head = concepts[heads[idx]]
            tail = concepts[tails[idx]]
            _rel = []
            for r in rel:
                r = self.preprocess([head]+r.split()+[tail], clause=True)
                _rel.append(r)
            _relations.append(_rel.copy())
        graph["relations"] = _relations
        concepts = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                    concepts]
        graph["concepts"] = concepts

        return graph

def collate_fn(data):
    def merge(sequences, scores=None, positions=None):
        """
        padded_seqs: use 1 to pad the rest
        lengths: the lengths of seq in sequences
        """
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long()  ## padding index 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x["context"]), reverse=True)  ## sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    input_batch, input_lengths = merge(item_info['context'])
    mask_input, mask_input_lengths = merge(item_info['context_mask'])  # use idx for bot or user to mask the seq
    causepos, _ = merge(item_info['causepos'])

    ## clause
    clause_batch, _ = merge(item_info["clause"])

    ## Target
    target_batch, target_lengths = merge(item_info['target'])

    if config.USE_CUDA:
        input_batch = input_batch.cuda()
        mask_input = mask_input.cuda()
        target_batch = target_batch.cuda()
        causepos = causepos.cuda()


    d = {}
    d["input_batch"] = input_batch
    d["input_lengths"] = torch.LongTensor(input_lengths)  # mask_input_lengths equals input_lengths
    d["causepos"] = causepos
    d["mask_input"] = mask_input
    ##cause
    d["cause_batch"] = item_info['cause_batch']
    d["botcause_clause"] = clause_batch
    d["botcause_label"] = item_info['botcause_label']
    ##target
    d["target_batch"] = target_batch
    d["target_lengths"] = torch.LongTensor(target_lengths)
    ##program
    d["target_program"] = item_info['emotion']  # one hot format
    d["program_label"] = item_info['emotion_label']
    ##graph
    d["graphs"] = item_info['graphs']
    ##text
    d["input_txt"] = item_info['context_text']
    d["target_txt"] = item_info['target_text']
    d["program_txt"] = item_info['emotion_text']

    return d

def prepare_data_seq(batch_size=32):
    """
    :return:
    vocab: vocabulary including index2word, and word2index
    len(dataset_train.emo_map)
    """
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()

    logging.info("Vocab  {} ".format(vocab.n_words))

    dataset_train = Dataset(pairs_tra, vocab)
    data_loader_tra = torch.utils.data.DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)

    dataset_valid = Dataset(pairs_val, vocab)
    data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid,
                              batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)

    dataset_test = Dataset(pairs_tst, vocab)
    data_loader_tst = torch.utils.data.DataLoader(dataset=dataset_test,
                              batch_size=1,
                              shuffle=False, collate_fn=collate_fn)
    write_config()
    return data_loader_tra, data_loader_val, data_loader_tst, vocab, len(dataset_train.emo_map)