
import json
from tqdm import tqdm
import networkx as nx
import spacy
nlp = spacy.load('en_core_web_md', disable=['ner', 'parser', 'textcat'])
import os

import config
from dataprocess.build_graph import load_resources

blacklist = {'-PRON-', 'whither', 'iq', 'al', 'xk', 'et-al', 'resulting', 'mg', 'specified', 'more', 'rf', 'c3', 'else', 'whence', 'usefulness', 'rr', 'est', 'made', 'edu', 'somehow', 'below', 'besides', 'thereby', 'thousand', 'ag', 'it', 'tp', 'lately', 'J', 'described', 'id', 'if', 'dp', 'want', 'e', 'L', 'arent', 'yl', 'sd', 'secondly', 'my', 'a3', 'already', 'f2', 'mostly', 'hardly', 'whenever', 'vs', 'qv', 'ra', 'we', 'bl', 'our', 'doing', 'added', 'hadn', 'hi', 'how', 'r', 'indicates', 'slightly', 'shed', 'always', 'anyways', 'right', 'ui', 'aside', 'page', 'considering', 'won', 'move', 'forty', 'ms', 'ey', 'whom', 'th', 'cv', 'its', 'both', 'probably', 'tc', 'hed', 'everything', 'pf', 'fi', 'keep', 'some', 'e3', 'with', 'entirely', 'somethan', 'td', 'av', '3a', 'is', 'comes', 'di', 'fu', 'la', 'might', 'cannot', 'up', 'follows', 'showed', 'trying', 'who', 'amount', 'usually', 'giving', 'cf', 'thereof', 'bottom', 'inasmuch', 'given', 'r2', 'insofar', 'ba', 'rather', 'hj', 'from', 'cr', 'latter', 'rd', 'respectively', 'though', 'anyway', 'kept', 'ho', 'til', 'P', 'wherever', 'vo', 'does', 'fify', 'quickly', 'ar', 'b1', 'W', 'cm', '6b', 'almost', 'e2', 'ip', 'theres', 'million', 'y', 'vols', 'ns', '6o', 'before', 'lr', 'they', 'why', 'dx', 'gj', 'research-articl', 'little', 'run', 'fa', 'ur', 'using', 'whomever', 'ih', 'which', 'whereafter', 'outside', 'xj', 'concerning', 'mainly', 'therere', 'yours', 'looking', 'thoroughly', 'cq', 'bt', 'for', 'gave', 'could_be', 'affected', 'six', 'obtain', 'somewhat', 'mn', 'eight', 'hereby', 'me', 'ran', 'b', 'ru', 'thickv', 'throughout', 'four', 'please', 'U', 'soon', 'tb', 'sixty', 'ob', 'successfully', 'km', 'necessarily', 'neither', 'show', 'shown', 'Z', 'about', 'H', 'a1', 'nn', 'dj', 'se', 'que', 'strongly', 'indeed', 'took', 'appreciate', 'i8', 'rn', 'us', 'ge', 'http', 'ou', 'sc', 'taken', 'og', 'hes', 'front', 'p2', 'top', 'want_to', 'pagecount', 'or', 'several', 'third', 'lb', 'specify', 'two', 'op', 'each', 'sincere', 'os', 'alone', 'yr', 'side', 'let', 'nr', 'anyone', 'ep', 'gets', 'mug', 'pas', 'presumably', 'y2', 'sometimes', 'able', 'oz', 'useful', 'when', 'old', 'dl', 'ri', 'got', 'someone', 'oo', 'obviously', 'pe', 'cc', 'latterly', 'theirs', 'beyond', 'B', 'own', 'as', 'welcome', 'oq', 'suggest', 'xi', 'while', 'va', 'bs', 'whether', 'following', 'non', 'second', 'eq', 'zero', 'ev', 's2', 'ph', 'fs', 'pc', 'no', 'wed', 'hereupon', 'ord', 'most', 'par', 'viz', 'xs', '0o', 'rh', 'themselves', 'ci', 'still', 'ts', 't3', 'toward', 'shes', 'E', 'regardless', 'give', 'just', 'zi', 'on', 'but', 'sn', 'mo', 'another', 'way', 'done', 'goes', 'ain', 'ec', 'couldn', 'forth', 'po', 'uk', 'none', 'bill', 'exactly', 'io', 'proud', 'cn', 'eleven', 'between', 'part', 'rj', 'once', 's', 'D', 'although', 'ln', 'vol', 'volumtype', 'now', 'tl', 'unlike', 'mustn', 'm', 'name', 'nos', 'l2', 'ca', 'itd', 'hs', 'three', 'o', 'that', 'seeming', 'afterwards', 'ft', 'ic', 'shows', 'through', 'anymore', 'don', 'thanks', 'gs', 'oc', 'among', 'pt', 'yes', 'didn', 'w', 'towards', 'whatever', 'whos', 'substantially', 'next', 'recent', 'sent', 'thorough', 'ma', 'any', 'eg', 'according', 'howbeit', 'beginnings', 'and', 'by', 'vt', 'thereupon', 'shall', 'ny', 'far', 'fo', 'hence', 'fn', 'ac', 'whose', 'may', 'fl', 'sec', 'says', 'www', 'near', 'cg', 'herein', 'cd', 'recently', 'tends', 'da', 'heres', 'clearly', 'ib', 'over', 'wheres', 'xn', 'js', 'those', 'many', 'widely', 'used', 'went', 'br', 'regarding', 'nevertheless', 'until', 'ao', 'particular', 'ref', 'ue', 'hello', 'ad', 'found', 'qj', 'relatively', 'l', 'per', 'p3', 'came', 'es', 'moreover', 'away', 'affecting', 'nt', 'sufficiently', 'various', 'ir', 'bk', '0s', 'regards', 'sometime', 'truly', 'hy', 'xo', 'i6', 'ten', 'ti', 'ask', 'related', 'thanx', 'apart', 'com', 'jr', 'xl', 'gone', 'dy', 'iy', 'tr', 'pp', 'section', 'aw', 'said', 'sure', 'tell', 'unfortunately', 'best', 'information', 'pi', 'sz', 'under', 'this', 'hasnt', 'whoever', 'fire', 'less', 'whod', 'ch', 'n', 'tx', 'tv', 'index', 'lj', 'nonetheless', 'sl', 'your', 'did', 'had', 'whole', 'hu', 'ought', 'owing', 'going', 'keeps', 'nd', 'so', 'few', 'ah', 'gotten', 'not', 'thou', 'anybody', 'lest', 'lf', 'tried', 'consequently', 'wouldnt', 'refs', 'yt', 'be', 'well', 'v', 'n2', 'rq', 'in', 'get', 'ga', 'you', 'something', 'every', 'became', 'ay', 'sy', 'au', 'az', 'co', 'st', 'i4', 'thereto', 'df', 'then', 'thoughh', 'ibid', 'pq', 'rc', 'm2', 'actually', 'meantime', 'help', 'bc', 'bu', 'consider', 'seems', 'saying', 'cry', 'fy', 'ninety', 'too', 'can', 'getting', 'otherwise', 'et', 'tip', 'fifth', 'ox', 'ei', 't2', 'haven', 'make', 'aren', 'has', 'indicated', 'interest', 'very', 'previously', 'tries', 'mrs', 'hid', 'course', 'followed', 'except', 'oa', 'ej', 'primarily', 'los', 'thence', 'S', 'sa', 'downwards', 'amoungst', 'doesn', 'twice', 'z', 'xv', 'detail', 'full', 'hh', 'above', '3b', 'ed', 'oj', 'af', 'date', 'usefully', 'youd', 'sf', 'bi', 'must', 'whim', 'ok', 'ef', 'ps', 'anyhow', 'nor', 'former', 'bn', 'ce', 'sj', 'j', 'er', 'sup', 'novel', 'off', 'du', 'anywhere', 'bd', 'predominantly', 'ss', '3d', 'fc', 'liked', 'rm', 'upon', 'resulted', 'take', 'oi', 'cj', 'c', 'largely', 'p', 'ending', 'put', 'b2', 'dt', 'even', 'mu', 'unless', 'really', 'thus', 'X', 'after', 'gl', 'k', 'nl', 'therein', 'again', 'are', 'ex', 'nj', 'lets', 'ones', 'sm', 'iv', 'C', 'end', 'via', 'omitted', 'cl', 'inner', 'h3', 'looks', 'around', 'M', 'seemed', 'couldnt', 'together', 'od', 'somewhere', 'cu', 'despite', 'would', 'hopefully', 'thered', 'eu', 'om', 'cp', 'un', 'been', 'nine', 'showns', 'ia', 'five', 'im', 'cant', 'apparently', 'well-b', 'ko', 'isn', 'the', 'specifying', 'especially', 'certainly', 'd', 'ea', 'particularly', 'R', 'taking', 'ff', 'furthermore', 've', 'twelve', 'yj', 'where', 'beforehand', 'results', 'seven', 'their', 'eo', 'adj', 'i2', 'amongst', 'than', 'pd', 'sp', 'ae', 'lt', 'noted', 'out', 'readily', 'makes', 'okay', 'weren', 'h2', 'wouldn', 'cx', 'rs', 'con', 'call', 'a4', 'K', 'somebody', 'am', 'approximately', 'pages', 'mr', 'mt', 'nobody', 'lc', 'wi', 'shan', 'ix', 'tf', 'sorry', 'les', 'en', 'qu', 'd2', 'dd', 'past', 'em', 'thank', 'u201d', 'what', 'sub', 'um', 'dc', 'ee', 'jj', 'perhaps', 'tq', 'wasn', 'x1', 'stop', 'fix', 'etc', 'sometimes_people', 'could', 'O', 'ro', 'ups', 'xf', 'h', 'i7', 'fifteen', 'meanwhile', 'seem', 'reasonably', 'either', 'c1', 'thereafter', 'vj', 'tt', 'behind', 'sr', 'de', 'hasn', 'last', 'nearly', 'try', 'ut', 'were', 'wherein', 'gives', 'bx', 'everyone', 'mine', 'jt', 'x', 'si', 'p1', 'across', 'ls', 'beside', 'like', 'gi', 'zz', 'a2', 'mill', 'cs', 'since', 'whats', 'pr', 't', 'G', 'kj', 'only', 'fj', 'therefore', 'was', 'certain', 'll', 'look', 'eighty', 'during', 'b3', 'have', 'wont', 'back', 'bp', 'later', 'elsewhere', 'hither', 'ke', 'maybe', 'ry', 'namely', 'biol', 'dr', 'inward', 'provides', 'dk', 'promptly', 'ignored', 'specifically', 'pk', 'ow', 'thats', 'there', 'tn', 'further', 'cit', 'fr', 'announce', 'everywhere', 'accordingly', 'these', 'el', 'into', 'i3', 'u', 'wonder', 'ne', 'enough', 'along', 'unlikely', 'wo', 'of', 'werent', 'awfully', 'f', 'i', 'a', 'line', 'ii', 'vd', 'he', 'whereby', 'xt', 'ys', 'twenty', 'theyd', 'much', 'say', 'abst', 'fill', 'placed', 'ij', 'merely', 'pn', 'find', 'ng', 'possibly', 'pu', 'within', 'at', 'available', 'ltd', 'aj', 'bj', 'na', 're', 'ol', 'theyre', 'instead', 'pj', 'overall', 'cy', 'py', 'likely', 'also', 'ab', 'kg', 'ot', 'them', 'here', 'tj', 'mightn', 'A', 'vu', 'Q', 'rt', 'nc', 'yet', 'often', 'ct', 'x2', 'happens', 'pl', 'quite', 'come', 'rv', 'auth', 'allow', 'miss', 'N', 'ju', 'youre', 'throug', 'to', 'nowhere', 'uj', 'whereas', 'xx', 'such', 'down', 'gr', 'te', 'go', 'rl', 'hr', 'all', 't1', 'whereupon', 'act', 'wa', 'ds', 'everybody', 'F', 'x3', 'nay', 'onto', 'however', 'normally', 'il', 'indicate', 'without', 'noone', 'V', 'inc', 'sq', 'oh', 'thin', 'tm', 'describe', 'iz', 'ever', 'le', 'lo', 'ml', 'wasnt', 'greetings', 'obtained', 'allows', 'against', 'think', 'pm', 'brief', 'g', 'ni', 'vq', 'having', 'immediately', 'Y', 'saw', 'unto', 'ig', 'accordance', 'example', 'ap', 'briefly', 'one', 'cz', 'asking', 'new', 'c2', 'hundred', 'do', 'T', 'least', 'ie', 'arise', 'definitely', 'seen', 'uo', 'poorly', 'hereafter', 'thru', 'formerly', 'q', 'due', 'an', 'ax', 'gy', 'plus'}

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
            if t.pos_ == "NOUN" or t.pos_ == "VERB" or t.pos_ == "ADJ" or t.pos_ == "ADV":
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