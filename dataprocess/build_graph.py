import pandas as pd
from tqdm import tqdm

from networkx import MultiDiGraph, write_gpickle

import nltk
# nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')
nltk_stopwords += ["gone", "did", "going", "would", "could", "get", "in", "up", "may", "uk", "us", "take", "make", "object", "person", "people"]

import config

def load_resources():
    concept2id, id2concept, relation2id, id2relation = dict(), dict(), dict(), dict()

    with open(config.concept_vocab, 'r', encoding='UTF8') as f:
        for w in f.readlines():
            concept2id[w.strip()] = len(concept2id)
            id2concept[len(id2concept)] = w.strip()
    print('finish loading concept2id')

    with open(config.concept_rel, 'r', encoding='UTF8') as f:
        for rel in f.readlines():
            relation2id[rel.strip()] = len(relation2id)
            id2relation[len(id2relation)] = rel.strip()
    print('finish loading relation2id')

    return concept2id, relation2id, id2relation, id2concept

def save_net():
    concept2id, relation2id, id2relation, id2concept = load_resources()
    graph = MultiDiGraph()
    conceptnet = pd.read_csv(config.conceptnet, sep='\t', encoding='UTF8') #, header=['start', 'end', 'rel', 'weight'])

    for idx, line in tqdm(conceptnet.iterrows(), desc="saving to graph"):
        start, end, rel, weight = line
        if start == end or start in nltk_stopwords or end in nltk_stopwords:
            continue
        if str(rel) == 'HasContext':
            continue
        start = concept2id[str(start)]
        end = concept2id[str(end)]
        rel = relation2id[str(rel)]
        weight = float(weight)

        graph.add_edge(start, end, rel=rel, weight=weight)
        graph.add_edge(end, start, rel=rel+len(relation2id), weight=weight)

    write_gpickle(graph, config.conceptnet_graph)
