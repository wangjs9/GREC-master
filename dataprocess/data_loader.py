import torch
import torch.utils.data as data
import logging
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
        graphIdx = self.data["graphidx"][str(index)]

        graphs = self.preprocess_graph([self.data["graphs"][id] for id in graphIdx])
        item["graph_num"] = len(graphIdx)
        item["graph_concept_ids"], item["graph_concept_label"], item["graph_distances"], item["graph_relation"], \
        item["graph_head"], item["graph_tail"], item["graph_triple_label"], item["vocab_map"], item["map_mask"] = graphs

        item["context"], item["context_mask"] = self.preprocess((item["situation_text"], item["dialog_text"]))
        item["cause_batch"] = [self.preprocess(["CLS"]+cause, clause=True) for cause in item["cause_text"] if cause]
        item["context_text"] = item["situation_text"] + [ele for lst in item["dialog_text"] for ele in lst]
        item["clause"] = []
        item["causepos"] = [0]
        for num, text in enumerate(item["context_text"]):
            clause = self.preprocess(text, clause=True)
            item["clause"].append(clause)
            score = 0
            for i, label in enumerate(item["cause_label"]):
                try:
                    score += label[num]
                except IndexError:
                    continue
            for i in range(len(clause)):
                item["causepos"].append(score)
        max_score = max(item["causepos"])
        item["causepos"] = torch.LongTensor([max_score-score if score else 0 for score in item["causepos"]])

        item["target"] = self.preprocess([w for lis in item["target_text"] for w in lis]+["EOS"], clause=True)
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

    def preprocess_graph(self, graphs):
        G_concept_ids, G_concept_label, G_distance, G_relation, G_head, G_tail, G_triple_label = [], [], [], [], [], [], []
        vocab_maps = []
        map_mask = [0 for i in range(self.vocab.n_words)]
        for idx, graph in enumerate(graphs):
            concepts = graph["concepts"]
            vocab_map = []
            for w, idx in self.vocab.word2index.items():
                try:
                    pos = concepts.index(w)
                    vocab_map.append(pos)
                    map_mask[idx] = 1
                except ValueError:
                    vocab_map.append(0)
            relations = graph["relations"]
            G_relation.append(torch.LongTensor([r[0] for r in relations]))

            concepts = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                    concepts]
            G_concept_ids.append(torch.LongTensor(concepts))
            G_distance.append(torch.LongTensor(graph["distances"]))
            G_head.append(torch.LongTensor(graph["head_ids"]))
            G_tail.append(torch.LongTensor(graph["tail_ids"]))
            G_triple_label.append(torch.LongTensor(graph["triple_labels"]))
            G_concept_label.append(torch.LongTensor(graph["labels"]))

            vocab_maps.append(torch.LongTensor(vocab_map))

        return G_concept_ids, G_concept_label, G_distance, G_relation, G_head, G_tail, G_triple_label, vocab_maps, torch.LongTensor(map_mask)

def collate_fn(data):
    def merge(sequences, single=True, pad=1):
        """
        padded_seqs: use 1 to pad the rest
        lengths: the lengths of seq in sequences
        """
        if single:
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.ones(len(sequences), max(lengths)).long()  ## padding index 1
            padded_seqs *= pad
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths
        else:
            doc_lengths = [len(doc) for doc in sequences]
            seq_lengths = [max([len(seq) for seq in doc]) if len(doc) else 0 for doc in sequences]
            padded_seqs = torch.ones(len(sequences), max(doc_lengths), max(seq_lengths)).long()
            padded_seqs *= pad
            for i, doc in enumerate(sequences):
                for j, seq in enumerate(doc):
                    end = len(seq)
                    padded_seqs[i, j, :end] = seq[:end]
            lengths = [[len(seq) for seq in doc]+[0 for i in range(max(doc_lengths)-len(doc))] for doc in sequences]
            return padded_seqs, lengths

    data.sort(key=lambda x: len(x["context"]), reverse=True)  ## sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## graph
    concept_ids, concept_num = merge(item_info["graph_concept_ids"], single=False)
    distances, _ = merge(item_info["graph_distances"], single=False, pad=0)
    relations, triple_num = merge(item_info["graph_relation"], single=False)
    heads, _ = merge(item_info["graph_head"], single=False)
    tails, _ = merge(item_info["graph_tail"], single=False)
    triple_label, _ = merge(item_info["graph_triple_label"], single=False, pad=-1)
    concept_label, _ = merge(item_info["graph_concept_label"], single=False, pad=-1)
    vocab_map, _ = merge(item_info["vocab_map"], single=False, pad=0)
    map_mask, _ = merge(item_info["map_mask"], pad=0)


    ## input
    input_batch, input_lengths = merge(item_info['context'])
    mask_input, mask_input_lengths = merge(item_info['context_mask'])  # use idx for bot or user to mask the seq
    causepos, _ = merge(item_info['causepos'])

    ## clause
    cause_batch, _ = merge(item_info['cause_batch'], single=False)
    clause_batch, _ = merge(item_info["clause"], single=False)

    ## Target
    target_batch, target_lengths = merge(item_info['target'])

    d = {}
    d["input_batch"] = input_batch.to(config.device)
    d["input_lengths"] = torch.LongTensor(input_lengths)  # mask_input_lengths equals input_lengths
    d["causepos"] = causepos.to(config.device)
    d["mask_input"] = mask_input.to(config.device)
    ##cause
    d["cause_batch"] = cause_batch.to(config.device)
    d["botcause_clause"] = clause_batch.to(config.device)
    d["botcause_label"] = item_info['botcause_label']
    ##target
    d["target_batch"] = target_batch.to(config.device)
    d["target_lengths"] = torch.LongTensor(target_lengths)
    ##program
    d["target_program"] = item_info['emotion']  # one hot format
    d["program_label"] = item_info['emotion_label']
    ##graph
    d["concept_ids"] = concept_ids.to(config.device)
    d["concept_num"] = concept_num
    d["distances"] = distances.to(config.device)
    d["relations"] = relations.to(config.device)
    d["triple_num"] = triple_num
    d["heads"] = heads.to(config.device)
    d["tails"] = tails.to(config.device)
    d["concept_label"] = concept_label.to(config.device)
    d["triple_label"] = triple_label.to(config.device)
    d["graph_num"] = item_info["graph_num"]
    d["vocab_map"] = vocab_map
    d["map_mask"] = map_mask
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
    # return data_loader_tra, None, None, vocab, len(dataset_train.emo_map)

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