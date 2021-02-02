import json
from tqdm import tqdm

def read_json(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def directed_triple(data_path, save_path, max_concepts=400, max_triple=1000):
    # read data from data_path
    data = read_json(data_path)

    _data = []
    max_len = 0
    max_neighbors = 5
    for e in tqdm(data):
        triple_dict = {}
        triples = e['triples']
        concepts = e['concepts']
        labels = e['labels']
        distances = e['distances']
        for t in triples:
            head, tail = t[0], t[-1]
            head_id = concepts.index(head)
            tail_id = concepts.index(tail)
            if distances[head_id] <= distances[tail_id]:
                if t[-1] not in triple_dict:
                    triple_dict[t[-1]] = [t]
                else:
                    if len(triple_dict[t[-1]]) < max_neighbors:
                        triple_dict[t[-1]].append(t)

        results = []
        for l, c in zip(labels, concepts):
            if l == 1:
                results.append(c)
        # print(results)

        causes = []
        for d, c in zip(distances, concepts):
            if d == 0:
                causes.append(c)

        shortest_paths = []
        for result in results:
            shortest_paths.extend(bfs(result, triple_dict, causes))

        ground_truth_concepts = []
        ground_truth_triples = []
        for path in shortest_paths:
            for i, n in enumerate(path[:-1]):
                ground_truth_triples.append((n, path[i + 1]))
                ground_truth_concepts.append(n)
                ground_truth_concepts.append(path[i + 1])
        ground_truth_concepts = list(set(ground_truth_concepts))

        ground_truth_triples_set = set(ground_truth_triples)

        _triples, triple_labels = [], []
        for e1, e2 in ground_truth_triples_set:
            for t in triple_dict[e1]:
                if e2 in t:
                    _triples.append(t)
                    triple_labels.append(1)

        for k, v in triple_dict.items():
            for t in v:
                if t in _triples:
                    continue
                _triples.append(t)
                # if (t[-1], t[0]) in ground_truth_triples_set:
                #     triple_labels.append(1)
                # else:
                triple_labels.append(0)

        if len(concepts) > max_concepts:
            rest_concepts = list(set(concepts) - set(ground_truth_concepts))
            rest_len = max_concepts-len(ground_truth_concepts)
            _concepts = ground_truth_concepts + rest_concepts[:rest_len]
            e['concepts'] = _concepts
            e['distances'] = [distances[concepts.index(c)] for c in _concepts]
            e['labels'] = [distances[labels.index(c)] for c in _concepts]
            concepts = _concepts
        # _triples = _triples[:max_triples]
        # triple_labels = triple_labels[:max_triples]

        heads = []
        tails = []
        relations = []
        for triple in _triples:
            try:
                h = concepts.index(triple[0])
                t = concepts.index(triple[-1])
                heads.append(h)
                tails.append(t)
                relations.append(triple[1])
                if len(heads) == max_triple:
                    break
            except ValueError:
                continue

        max_len = max(max_len, len(_triples))
        e['relations'] = relations
        e['head_ids'] = heads
        e['tail_ids'] = tails
        e['triple_labels'] = triple_labels[:max_triple]
        e.pop('triples')

        _data.append(e)
        # break

    with open(save_path, 'w') as f:
        for line in data:
            json.dump(line, f)
            f.write('\n')

    return _data

def bfs(start, triple_dict, source):
    paths = [[[start]]]
    shortest_paths = []
    count = 0
    while True:
        last_paths = paths[-1]
        new_paths = []
        for path in last_paths:
            if triple_dict.get(path[-1], False):
                triples = triple_dict[path[-1]]
                for triple in triples:
                    new_paths.append(path + [triple[0]])

        for path in new_paths:
            if path[-1] in source:
                shortest_paths.append(path)

        if count == 2:
            break
        paths.append(new_paths)
        count += 1

    return shortest_paths