
import json
from tqdm import tqdm
import networkx as nx
import spacy
nlp = spacy.load('en_core_web_md', disable=['ner', 'parser', 'textcat'])
import os

import config
from dataprocess.build_graph import load_resources

blacklist = set(["from", "as", "more", "either", "in", "and", "on", "an", "when", "too", "to", "i", "do", "can", "be", "that", "or", "the", "a", "of", "for", "is", "was", "the", "-PRON-", "actually", "likely", "possibly", "want",
                 "make", "my", "someone", "sometimes_people", "sometimes","would", "want_to", "one", "something", "sometimes", "everybody", "somebody", "could", "could_be","mine","us","em",
                 "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "A", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "after", "afterwards", "ag", "again", "against", "ah", "ain", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appreciate", "approximately", "ar", "are", "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "B", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "been", "before", "beforehand", "beginnings", "behind", "below", "beside", "besides", "best", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "C", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "ci", "cit", "cj", "cl", "clearly", "cm", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "could", "couldn", "couldnt", "course", "cp", "cq", "cr", "cry", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d", "D", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "dj", "dk", "dl", "do", "does", "doesn", "doing", "don", "done", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "E", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "F", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "G", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "H", "h2", "h3", "had", "hadn", "happens", "hardly", "has", "hasn", "hasnt", "have", "haven", "having", "he", "hed", "hello", "help", "hence", "here", "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hh", "hi", "hid", "hither", "hj", "ho", "hopefully", "how", "howbeit", "however", "hr", "hs", "http", "hu", "hundred", "hy", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "im", "immediately", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "inward", "io", "ip", "iq", "ir", "is", "isn", "it", "itd", "its", "iv", "ix", "iy", "iz", "j", "J", "jj", "jr", "js", "jt", "ju", "just", "k", "K", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "ko", "l", "L", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "M", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "my", "n", "N", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "neither", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "O", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "otherwise", "ou", "ought", "our", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "P", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "pp", "pq", "pr", "predominantly", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "Q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "R", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "S", "s2", "sa", "said", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "seem", "seemed", "seeming", "seems", "seen", "sent", "seven", "several", "sf", "shall", "shan", "shed", "shes", "show", "showed", "shown", "showns", "shows", "si", "side", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somehow", "somethan", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "sz", "t", "T", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "thereof", "therere", "theres", "thereto", "thereupon", "these", "they", "theyd", "theyre", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "U", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "used", "useful", "usefully", "usefulness", "using", "usually", "ut", "v", "V", "va", "various", "vd", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "W", "wa", "was", "wasn", "wasnt", "way", "we", "wed", "welcome", "well", "well-b", "went", "were", "weren", "werent", "what", "whatever", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "whom", "whomever", "whos", "whose", "why", "wi", "widely", "with", "within", "without", "wo", "won", "wonder", "wont", "would", "wouldn", "wouldnt", "www", "x", "X", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "Y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "your", "youre", "yours", "yr", "ys", "yt", "z", "Z", "zero", "zi", "zz"])

def load_cpnet():
    global concept2id, relation2id, id2relation, id2concept, concept_vocab
    global cpnet, cpnet_simple

    def jsonKeys2int(x):
        if isinstance(x, dict):
            return {int(k): v for k, v in x.items()}
        return x

    if os.path.exists(config.data_concept_dict+'concept_info.json'):
        with open(config.data_concept_dict+'concept_info.json', 'r') as f:
            data = f.readlines()
            concept2id = json.loads(data[0])
            relation2id = json.loads(data[1])
            id2relation = json.loads(data[2], object_hook=jsonKeys2int)
            id2concept = json.loads(data[3], object_hook=jsonKeys2int)

    else:
        concept2id, relation2id, id2relation, id2concept = load_resources()
        with open(config.data_concept_dict+'concept_info.json', 'w') as f:
            json.dump(concept2id, f)
            f.write('\n')
            json.dump(relation2id, f)
            f.write('\n')
            json.dump(id2relation, f)
            f.write('\n')
            json.dump(id2concept, f)
            f.write('\n')

    concept_vocab = concept2id.keys()
    cpnet = nx.read_gpickle(config.conceptnet_graph)

    if os.path.exists(config.data_concept_dict+'simple_concept.graph'):
        cpnet_simple = nx.read_gpickle(config.data_concept_dict+'simple_concept.graph')

    else:
        cpnet_simple = nx.Graph()
        for u, v, data in cpnet.edges(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            if cpnet_simple.has_edge(u, v):
                cpnet_simple[u][v]['weight'] += w
            else:
                cpnet_simple.add_edge(u, v, weight=w)
        nx.write_gpickle(cpnet_simple, config.data_concept_dict+'simple_concept.graph')

def hard_ground(sent):
    doc = nlp(sent)
    res = set()
    for t in doc:
        if t.lemma_ not in blacklist:
            if t.pos_ == "NOUN" or t.pos_ == "VERB" :
                if t.lemma_ in concept_vocab:
                    res.add(t.lemma_)
                if t.pos_ == "NOUN" and t.text in concept_vocab:
                   res.add(t.text)
    return res

def match_concepts(start, end):
    res = []
    total_concepts = set()
    global total_concepts_id
    total_concepts_id = set(total_concepts_id)
    for sid, s in tqdm(enumerate(start), total=len(start)):
        e = end[sid]
        start_concepts = hard_ground(s)
        end_concepts = hard_ground(e) - start_concepts
        total_concepts.update(start_concepts)
        total_concepts.update(end_concepts)
        res.append({'start': list(start_concepts), 'end': list(end_concepts)})

    total_concepts_id.update(set([concept2id[w] for w in total_concepts]))
    total_concepts_id = sorted(list(total_concepts_id))
    return res

def find_neighbours_frequency(source_concepts, target_concepts, T, max_B=50, max_search=12):
    start = [concept2id[s_cpt] for s_cpt in source_concepts]  # start nodes for each turn
    Vts = dict([(x, 0) for x in start])  # nodes and their turn

    if len(target_concepts) == 0 or len(source_concepts) == 0:
        return {"concepts": [], "labels": [], "distances": [], "triples": []}, -1, 0

    Ets = {} # nodes and their neighbor in last turn
    target_concpet_nlp = [nlp(t_cpt)[0] for t_cpt in target_concepts]
    ts = [concept2id[t_cpt] for t_cpt in target_concepts]
    for t in range(T):
        V = {}
        for s in start:
            if s in cpnet_simple and s not in ts:
                candidates = [c for c in cpnet_simple[s] if c not in Vts and c in total_concepts_id]
                scores = {c: max([nlp(id2concept[c])[0].similarity(t_cpt) for t_cpt in target_concpet_nlp]) for c in candidates}
                candidates = sorted(list(scores.items()), key=lambda x: x[1], reverse=True)[:max_search]
                candidates = [x[0] for x in candidates]
                for c in candidates:
                    if c not in V:
                        V[c] = scores[c]
                    else:
                        V[c] += scores[c]
                    rels = get_edge(s, c)
                    if len(rels) > 0:
                        if c not in Ets:
                            Ets[c] = {s: rels}
                        else:
                            Ets[c].update({s: rels})

        V = list(V.items())
        count_V = sorted(V, key=lambda x: x[1], reverse=True)[:max_B] # the top max_B nodes related to entities in start concepts
        start = [x[0] for x in count_V] # filter the nodes excluded from the dataset

        Vts.update(dict([(x, t + 1) for x in start])) # add new nodes

    concepts = list(Vts.keys())
    distances = list(Vts.values())
    assert (len(concepts) == len(distances))

    triples = []

    for v, N in Ets.items():
        if v in concepts:
            for u, rels in N.items():
                # if u in concepts:
                triples.append((u, rels, v))

    labels = []
    found_num = 0
    for c in concepts:
        if c in ts:
            found_num += 1
            labels.append(1)
        else:
            labels.append(0)

    res = [id2concept[x].replace("_", " ") for x in concepts]
    triples = [(id2concept[x].replace("_", " "), y, id2concept[z].replace("_", " ")) for (x, y, z) in triples]
    # print(found_num)
    return {"concepts": res, "labels": labels, "distances": distances, "triples": triples}, found_num, len(res)

def get_edge(src_concept, tgt_concept):
    try:
        rel_list = cpnet[src_concept][tgt_concept]
        return list(set([rel_list[item]['rel'] for item in rel_list]))
    except:
        return []

def process(save_path, start, end, idlist, T, max_B=25):
    load_cpnet()
    print('Generating concept version ...')
    global total_concepts_id
    total_concepts_id = []
    if os.path.exists(config.data_concept_dict + 'total_concept_id.txt'):
        with open(config.data_concept_dict + 'total_concept_id.txt', 'r') as f:
            data = f.readlines()
            total_concepts_id = [int(line.strip()) for line in data]

    mode = save_path.split('.')[-2]
    if os.path.exists(config.data_concept_dict + 'sys_concepts.{}.json'.format(mode)):
        conceptVer = []
        with open(config.data_concept_dict + 'sys_concepts.{}.json'.format(mode), 'r') as f:
            data = f.readlines()
        for line in data:
            conceptVer.append(json.loads(line))

    else:
        conceptVer = match_concepts(start, end)
        with open(config.data_concept_dict + 'sys_concepts.{}.json'.format(mode), 'w') as f:
            for line in conceptVer:
                json.dump(line, f)
                f.write('\n')
        with open(config.data_concept_dict + 'total_concept_id.txt', 'w') as f:
            for line in total_concepts_id:
                f.write(str(line))
                f.write('\n')

    print('Done')
    examples = []
    avg_len = 0
    avg_found = 0
    total_vaild = 0
    print('Finding paths ...')
    for pair in tqdm(conceptVer):
        info, found, avg_nodes = find_neighbours_frequency(pair['start'], pair['end'], T, max_B)
        avg_len += avg_nodes
        if found != -1:
            avg_found += found
            total_vaild += 1
        examples.append(info)
    print('{} hops avg nodes: {} avg_path: {}'.format(T, avg_len / len(examples), avg_found/total_vaild))

    newIdlist, tmpidx = {}, []
    curId, count = 0, 0
    with open(save_path, 'w') as f:
        for i, line in enumerate(examples):
            if idlist[i] != curId:
                newIdlist[curId] = tmpidx
                curId = idlist[i]
                tmpidx = []
            if line["concepts"] == []:
                continue
            tmpidx.append(count)
            count += 1
            json.dump(line, f)
            f.write('\n')
        newIdlist[curId] = tmpidx

    return newIdlist