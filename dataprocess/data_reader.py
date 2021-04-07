import config
import numpy as np
import nltk
import os, pickle, json
from models.cause_extraction import predict
from dataprocess.find_path import process
from dataprocess.triple_store import directed_triple, read_json

blacklist = {'-PRON-', 'whither', 'iq', 'al', 'xk', 'et-al', 'resulting', 'mg', 'specified', 'more', 'rf', 'c3', 'else', 'whence', 'usefulness', 'rr', 'est', 'made', 'edu', 'somehow', 'below', 'besides', 'thereby', 'thousand', 'ag', 'it', 'tp', 'lately', 'J', 'described', 'id', 'if', 'dp', 'want', 'e', 'L', 'arent', 'yl', 'sd', 'secondly', 'my', 'a3', 'already', 'f2', 'mostly', 'hardly', 'whenever', 'vs', 'qv', 'ra', 'we', 'bl', 'our', 'doing', 'added', 'hadn', 'hi', 'how', 'r', 'indicates', 'slightly', 'shed', 'always', 'anyways', 'right', 'ui', 'aside', 'page', 'considering', 'won', 'move', 'forty', 'ms', 'ey', 'whom', 'th', 'cv', 'its', 'both', 'probably', 'tc', 'hed', 'everything', 'pf', 'fi', 'keep', 'some', 'e3', 'with', 'entirely', 'somethan', 'td', 'av', '3a', 'is', 'comes', 'di', 'fu', 'la', 'might', 'cannot', 'up', 'follows', 'showed', 'trying', 'who', 'amount', 'usually', 'giving', 'cf', 'thereof', 'bottom', 'inasmuch', 'given', 'r2', 'insofar', 'ba', 'rather', 'hj', 'from', 'cr', 'latter', 'rd', 'respectively', 'though', 'anyway', 'kept', 'ho', 'til', 'P', 'wherever', 'vo', 'does', 'fify', 'quickly', 'ar', 'b1', 'W', 'cm', '6b', 'almost', 'e2', 'ip', 'theres', 'million', 'y', 'vols', 'ns', '6o', 'before', 'lr', 'they', 'why', 'dx', 'gj', 'research-articl', 'little', 'run', 'fa', 'ur', 'using', 'whomever', 'ih', 'which', 'whereafter', 'outside', 'xj', 'concerning', 'mainly', 'therere', 'yours', 'looking', 'thoroughly', 'cq', 'bt', 'for', 'gave', 'could_be', 'affected', 'six', 'obtain', 'somewhat', 'mn', 'eight', 'hereby', 'me', 'ran', 'b', 'ru', 'thickv', 'throughout', 'four', 'please', 'U', 'soon', 'tb', 'sixty', 'ob', 'successfully', 'km', 'necessarily', 'neither', 'show', 'shown', 'Z', 'about', 'H', 'a1', 'nn', 'dj', 'se', 'que', 'strongly', 'indeed', 'took', 'appreciate', 'i8', 'rn', 'us', 'ge', 'http', 'ou', 'sc', 'taken', 'og', 'hes', 'front', 'p2', 'top', 'want_to', 'pagecount', 'or', 'several', 'third', 'lb', 'specify', 'two', 'op', 'each', 'sincere', 'os', 'alone', 'yr', 'side', 'let', 'nr', 'anyone', 'ep', 'gets', 'mug', 'pas', 'presumably', 'y2', 'sometimes', 'able', 'oz', 'useful', 'when', 'old', 'dl', 'ri', 'got', 'someone', 'oo', 'obviously', 'pe', 'cc', 'latterly', 'theirs', 'beyond', 'B', 'own', 'as', 'welcome', 'oq', 'suggest', 'xi', 'while', 'va', 'bs', 'whether', 'following', 'non', 'second', 'eq', 'zero', 'ev', 's2', 'ph', 'fs', 'pc', 'no', 'wed', 'hereupon', 'ord', 'most', 'par', 'viz', 'xs', '0o', 'rh', 'themselves', 'ci', 'still', 'ts', 't3', 'toward', 'shes', 'E', 'regardless', 'give', 'just', 'zi', 'on', 'but', 'sn', 'mo', 'another', 'way', 'done', 'goes', 'ain', 'ec', 'couldn', 'forth', 'po', 'uk', 'none', 'bill', 'exactly', 'io', 'proud', 'cn', 'eleven', 'between', 'part', 'rj', 'once', 's', 'D', 'although', 'ln', 'vol', 'volumtype', 'now', 'tl', 'unlike', 'mustn', 'm', 'name', 'nos', 'l2', 'ca', 'itd', 'hs', 'three', 'o', 'that', 'seeming', 'afterwards', 'ft', 'ic', 'shows', 'through', 'anymore', 'don', 'thanks', 'gs', 'oc', 'among', 'pt', 'yes', 'didn', 'w', 'towards', 'whatever', 'whos', 'substantially', 'next', 'recent', 'sent', 'thorough', 'ma', 'any', 'eg', 'according', 'howbeit', 'beginnings', 'and', 'by', 'vt', 'thereupon', 'shall', 'ny', 'far', 'fo', 'hence', 'fn', 'ac', 'whose', 'may', 'fl', 'sec', 'says', 'www', 'near', 'cg', 'herein', 'cd', 'recently', 'tends', 'da', 'heres', 'clearly', 'ib', 'over', 'wheres', 'xn', 'js', 'those', 'many', 'widely', 'used', 'went', 'br', 'regarding', 'nevertheless', 'until', 'ao', 'particular', 'ref', 'ue', 'hello', 'ad', 'found', 'qj', 'relatively', 'l', 'per', 'p3', 'came', 'es', 'moreover', 'away', 'affecting', 'nt', 'sufficiently', 'various', 'ir', 'bk', '0s', 'regards', 'sometime', 'truly', 'hy', 'xo', 'i6', 'ten', 'ti', 'ask', 'related', 'thanx', 'apart', 'com', 'jr', 'xl', 'gone', 'dy', 'iy', 'tr', 'pp', 'section', 'aw', 'said', 'sure', 'tell', 'unfortunately', 'best', 'information', 'pi', 'sz', 'under', 'this', 'hasnt', 'whoever', 'fire', 'less', 'whod', 'ch', 'n', 'tx', 'tv', 'index', 'lj', 'nonetheless', 'sl', 'your', 'did', 'had', 'whole', 'hu', 'ought', 'owing', 'going', 'keeps', 'nd', 'so', 'few', 'ah', 'gotten', 'not', 'thou', 'anybody', 'lest', 'lf', 'tried', 'consequently', 'wouldnt', 'refs', 'yt', 'be', 'well', 'v', 'n2', 'rq', 'in', 'get', 'ga', 'you', 'something', 'every', 'became', 'ay', 'sy', 'au', 'az', 'co', 'st', 'i4', 'thereto', 'df', 'then', 'thoughh', 'ibid', 'pq', 'rc', 'm2', 'actually', 'meantime', 'help', 'bc', 'bu', 'consider', 'seems', 'saying', 'cry', 'fy', 'ninety', 'too', 'can', 'getting', 'otherwise', 'et', 'tip', 'fifth', 'ox', 'ei', 't2', 'haven', 'make', 'aren', 'has', 'indicated', 'interest', 'very', 'previously', 'tries', 'mrs', 'hid', 'course', 'followed', 'except', 'oa', 'ej', 'primarily', 'los', 'thence', 'S', 'sa', 'downwards', 'amoungst', 'doesn', 'twice', 'z', 'xv', 'detail', 'full', 'hh', 'above', '3b', 'ed', 'oj', 'af', 'date', 'usefully', 'youd', 'sf', 'bi', 'must', 'whim', 'ok', 'ef', 'ps', 'anyhow', 'nor', 'former', 'bn', 'ce', 'sj', 'j', 'er', 'sup', 'novel', 'off', 'du', 'anywhere', 'bd', 'predominantly', 'ss', '3d', 'fc', 'liked', 'rm', 'upon', 'resulted', 'take', 'oi', 'cj', 'c', 'largely', 'p', 'ending', 'put', 'b2', 'dt', 'even', 'mu', 'unless', 'really', 'thus', 'X', 'after', 'gl', 'k', 'nl', 'therein', 'again', 'are', 'ex', 'nj', 'lets', 'ones', 'sm', 'iv', 'C', 'end', 'via', 'omitted', 'cl', 'inner', 'h3', 'looks', 'around', 'M', 'seemed', 'couldnt', 'together', 'od', 'somewhere', 'cu', 'despite', 'would', 'hopefully', 'thered', 'eu', 'om', 'cp', 'un', 'been', 'nine', 'showns', 'ia', 'five', 'im', 'cant', 'apparently', 'well-b', 'ko', 'isn', 'the', 'specifying', 'especially', 'certainly', 'd', 'ea', 'particularly', 'R', 'taking', 'ff', 'furthermore', 've', 'twelve', 'yj', 'where', 'beforehand', 'results', 'seven', 'their', 'eo', 'adj', 'i2', 'amongst', 'than', 'pd', 'sp', 'ae', 'lt', 'noted', 'out', 'readily', 'makes', 'okay', 'weren', 'h2', 'wouldn', 'cx', 'rs', 'con', 'call', 'a4', 'K', 'somebody', 'am', 'approximately', 'pages', 'mr', 'mt', 'nobody', 'lc', 'wi', 'shan', 'ix', 'tf', 'sorry', 'les', 'en', 'qu', 'd2', 'dd', 'past', 'em', 'thank', 'u201d', 'what', 'sub', 'um', 'dc', 'ee', 'jj', 'perhaps', 'tq', 'wasn', 'x1', 'stop', 'fix', 'etc', 'sometimes_people', 'could', 'O', 'ro', 'ups', 'xf', 'h', 'i7', 'fifteen', 'meanwhile', 'seem', 'reasonably', 'either', 'c1', 'thereafter', 'vj', 'tt', 'behind', 'sr', 'de', 'hasn', 'last', 'nearly', 'try', 'ut', 'were', 'wherein', 'gives', 'bx', 'everyone', 'mine', 'jt', 'x', 'si', 'p1', 'across', 'ls', 'beside', 'like', 'gi', 'zz', 'a2', 'mill', 'cs', 'since', 'whats', 'pr', 't', 'G', 'kj', 'only', 'fj', 'therefore', 'was', 'certain', 'll', 'look', 'eighty', 'during', 'b3', 'have', 'wont', 'back', 'bp', 'later', 'elsewhere', 'hither', 'ke', 'maybe', 'ry', 'namely', 'biol', 'dr', 'inward', 'provides', 'dk', 'promptly', 'ignored', 'specifically', 'pk', 'ow', 'thats', 'there', 'tn', 'further', 'cit', 'fr', 'announce', 'everywhere', 'accordingly', 'these', 'el', 'into', 'i3', 'u', 'wonder', 'ne', 'enough', 'along', 'unlikely', 'wo', 'of', 'werent', 'awfully', 'f', 'i', 'a', 'line', 'ii', 'vd', 'he', 'whereby', 'xt', 'ys', 'twenty', 'theyd', 'much', 'say', 'abst', 'fill', 'placed', 'ij', 'merely', 'pn', 'find', 'ng', 'possibly', 'pu', 'within', 'at', 'available', 'ltd', 'aj', 'bj', 'na', 're', 'ol', 'theyre', 'instead', 'pj', 'overall', 'cy', 'py', 'likely', 'also', 'ab', 'kg', 'ot', 'them', 'here', 'tj', 'mightn', 'A', 'vu', 'Q', 'rt', 'nc', 'yet', 'often', 'ct', 'x2', 'happens', 'pl', 'quite', 'come', 'rv', 'auth', 'allow', 'miss', 'N', 'ju', 'youre', 'throug', 'to', 'nowhere', 'uj', 'whereas', 'xx', 'such', 'down', 'gr', 'te', 'go', 'rl', 'hr', 'all', 't1', 'whereupon', 'act', 'wa', 'ds', 'everybody', 'F', 'x3', 'nay', 'onto', 'however', 'normally', 'il', 'indicate', 'without', 'noone', 'V', 'inc', 'sq', 'oh', 'thin', 'tm', 'describe', 'iz', 'ever', 'le', 'lo', 'ml', 'wasnt', 'greetings', 'obtained', 'allows', 'against', 'think', 'pm', 'brief', 'g', 'ni', 'vq', 'having', 'immediately', 'Y', 'saw', 'unto', 'ig', 'accordance', 'example', 'ap', 'briefly', 'one', 'cz', 'asking', 'new', 'c2', 'hundred', 'do', 'T', 'least', 'ie', 'arise', 'definitely', 'seen', 'uo', 'poorly', 'hereafter', 'thru', 'formerly', 'q', 'due', 'an', 'ax', 'gy', 'plus'}
class Lang:
    """
    create a new word dictionary, including 3 dictionaries:
    1) word to index;
    2) word and its count;
    3) index to word;
    and one counter indicating the number of words.
    """

    def __init__(self, init_index2word):
        """
        :param init_index2word: a dictionary containing (id: token) pairs
        """
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)  # Count default tokens

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def read_langs(vocab):
    # >>>>>>>>>> word pairs: replace some sentences in the paragraph >>>>>>>>>> #
    # >>>>>>>>>> historical utterances >>>>>>>>>> #
    train_dialog = np.load(config.data_npy_dict+'sys_dialog_texts.train.npy', allow_pickle=True)
    # >>>>>>>>>> next expected utterance from the bot >>>>>>>>>> #
    train_target = np.load(config.data_npy_dict+'sys_target_texts.train.npy', allow_pickle=True)
    # >>>>>>>>>> emotions of the conversation >>>>>>>>>> #
    train_emotion = np.load(config.data_npy_dict+'sys_emotion_texts.train.npy', allow_pickle=True)
    # >>>>>>>>>> prompts of the conversation >>>>>>>>>> #
    train_situation = np.load(config.data_npy_dict+'sys_situation_texts.train.npy', allow_pickle=True)
    necessary_file = ['usercause_texts', 'botcause_texts', 'usercause_labels', 'botcause_labels']
    for file in necessary_file:
        if not os.path.exists(config.data_npy_dict + 'sys_{}.train.npy'.format(file)):
            print('Generating cause clauses...')
            predict(train_situation, train_dialog, train_target, 'train', config.data_npy_dict)
            break
    # >>>>>>>>>> usercause of the conversation >>>>>>>>>> #
    train_usercause = np.load(config.data_npy_dict + 'sys_usercause_texts.train.npy', allow_pickle=True)
    # >>>>>>>>>> botcause of the conversation >>>>>>>>>> #
    # train_botcause = np.load(config.data_npy_dict + 'sys_botcause_texts.train.npy', allow_pickle=True)
    # >>>>>>>>>> usercause label of the conversation >>>>>>>>>> #
    train_usercause_label = np.load(config.data_npy_dict + 'sys_usercause_labels.train.npy',
                              allow_pickle=True)
    # >>>>>>>>>> botcause label of the conversation >>>>>>>>>> #
    # train_botcause_label = np.load(config.data_npy_dict + 'sys_botcause_labels.train.npy',allow_pickle=True)
    train_act_label = np.load(config.data_npy_dict + 'sys_act_texts.train.npy', allow_pickle=True)
    train_fake_same_act = np.load(config.data_npy_dict + 'sys_fake_same_act.train.npy', allow_pickle=True)
    train_fake_same_emo = np.load(config.data_npy_dict + 'sys_fake_same_emo.train.npy', allow_pickle=True)

    dev_dialog = np.load(config.data_npy_dict+'sys_dialog_texts.valid.npy', allow_pickle=True)
    dev_target = np.load(config.data_npy_dict+'sys_target_texts.valid.npy', allow_pickle=True)
    dev_emotion = np.load(config.data_npy_dict+'sys_emotion_texts.valid.npy', allow_pickle=True)
    dev_situation = np.load(config.data_npy_dict+'sys_situation_texts.valid.npy', allow_pickle=True)

    for file in necessary_file:
        if not os.path.exists(config.data_npy_dict + 'sys_{}.valid.npy'.format(file)):
            print('Generating cause clauses...')
            predict(dev_situation, dev_dialog, dev_target, 'valid', config.data_npy_dict)
    dev_usercause = np.load(config.data_npy_dict + 'sys_usercause_texts.valid.npy', allow_pickle=True)
    # dev_botcause = np.load(config.data_npy_dict + 'sys_botcause_texts.valid.npy', allow_pickle=True)
    dev_usercause_label = np.load(config.data_npy_dict + 'sys_usercause_labels.valid.npy', allow_pickle=True)
    # dev_botcause_label = np.load(config.data_npy_dict + 'sys_botcause_labels.valid.npy', allow_pickle=True)
    dev_act_label = np.load(config.data_npy_dict + 'sys_act_texts.valid.npy', allow_pickle=True)
    dev_fake_same_act = np.load(config.data_npy_dict + 'sys_fake_same_act.valid.npy', allow_pickle=True)
    dev_fake_same_emo = np.load(config.data_npy_dict + 'sys_fake_same_emo.valid.npy', allow_pickle=True)

    test_dialog = np.load(config.data_npy_dict+'sys_dialog_texts.test.npy', allow_pickle=True)
    test_target = np.load(config.data_npy_dict+'sys_target_texts.test.npy', allow_pickle=True)
    test_emotion = np.load(config.data_npy_dict+'sys_emotion_texts.test.npy', allow_pickle=True)
    test_situation = np.load(config.data_npy_dict+'sys_situation_texts.test.npy', allow_pickle=True)
    for file in necessary_file:
        if not os.path.exists(config.data_npy_dict + 'sys_{}.test.npy'.format(file)):
            print('Generating cause clauses...')
            predict(test_situation, test_dialog, test_target, 'test', config.data_npy_dict)
    test_usercause = np.load(config.data_npy_dict + 'sys_usercause_texts.test.npy', allow_pickle=True)
    # test_botcause = np.load(config.data_npy_dict + 'sys_botcause_texts.test.npy', allow_pickle=True)
    test_usercause_label = np.load(config.data_npy_dict + 'sys_usercause_labels.test.npy', allow_pickle=True)
    # test_botcause_label = np.load(config.data_npy_dict + 'sys_botcause_labels.test.npy'., allow_pickle=True)
    test_act_label = np.load(config.data_npy_dict + 'sys_act_texts.test.npy', allow_pickle=True)
    test_fake_same_act = np.load(config.data_npy_dict + 'sys_fake_same_act.test.npy', allow_pickle=True)
    test_fake_same_emo = np.load(config.data_npy_dict + 'sys_fake_same_emo.test.npy', allow_pickle=True)

    data_train = {'dialog': [], 'target': [], 'emotion': train_emotion, 'situation': [], 'usercause_label': train_usercause_label, 'usercause':[], 'graphs':[], 'graphidx': [], 'act_label': train_act_label, 'fake_same_act': train_fake_same_act, 'fake_same_emo': train_fake_same_emo}
    data_dev = {'dialog': [], 'target': [], 'emotion': [], 'situation': [], 'usercause_label': dev_usercause_label, 'usercause': [], 'graphs': [], 'graphidx': [], 'act_label': dev_act_label, 'fake_same_act': dev_fake_same_act, 'fake_same_emo': dev_fake_same_emo}
    data_test = {'dialog': [], 'target': [], 'emotion': [], 'situation': [], 'usercause_label': test_usercause_label, 'usercause': [], 'graphs': [], 'graphidx': [], 'act_label': test_act_label, 'fake_same_act': test_fake_same_act, 'fake_same_emo': test_fake_same_emo}

    for dialog in train_dialog:
        u_lists = []
        for utts in dialog:
            u_list = []
            for u in utts:
                u = nltk.word_tokenize(u)
                u_list.append(u)
                vocab.index_words(u)
            u_lists.append(u_list)
        data_train['dialog'].append(u_lists)

    for target in train_target:
        u_list = []
        for u in target:
            u = nltk.word_tokenize(u)

            u_list.append(u)
            vocab.index_words(u)

        data_train['target'].append(u_list)

    for situation in train_situation:
        u_list = []
        for u in situation:
            u = nltk.word_tokenize(u)
            u_list.append(u)
            vocab.index_words(u)
        data_train['situation'].append(u_list)

    for cause in train_usercause:
        u_lists = []
        for utts in cause:
            u_list =  nltk.word_tokenize(utts)
            u_lists.append(u_list)
        data_train['usercause'].append(u_lists)

    # for cause in train_botcause:
    #     u_lists = []
    #     for utts in cause:
    #         u_list = nltk.word_tokenize(utts)
    #         u_lists.append(u_list)
    #     data_train['botcause'].append(u_lists)


    assert len(data_train['dialog']) == len(data_train['target']) == len(data_train['emotion']) == len(
        data_train['situation']) == len(data_train['usercause']) == len(data_train['usercause_label'])

    for dialog in dev_dialog:
        u_lists = []
        for utts in dialog:
            u_list = []
            for u in utts:
                u = nltk.word_tokenize(u)
                u_list.append(u)
                vocab.index_words(u)
            u_lists.append(u_list)
        data_dev['dialog'].append(u_lists)

    for target in dev_target:
        u_list = []
        for u in target:
            u = nltk.word_tokenize(u)
            u_list.append(u)
            vocab.index_words(u)
        data_dev['target'].append(u_list)

    for situation in dev_situation:
        u_list = []
        for u in situation:
            u = nltk.word_tokenize(u)
            u_list.append(u)
            vocab.index_words(u)
        data_dev['situation'].append(u_list)

    for emotion in dev_emotion:
        data_dev['emotion'].append(emotion)

    for cause in dev_usercause:
        u_lists = []
        for utts in cause:
            u_list = nltk.word_tokenize(utts)
            u_lists.append(u_list)
        data_dev['usercause'].append(u_lists)

    # for cause in dev_botcause:
    #     u_lists = []
    #     for utts in cause:
    #         u_list = nltk.word_tokenize(utts)
    #         u_lists.append(u_list)
    #     data_dev['botcause'].append(u_lists)


    assert len(data_dev['dialog']) == len(data_dev['target']) == len(data_dev['emotion']) == len(
        data_dev['situation']) == len(data_dev['usercause']) == len(data_dev['usercause_label'])

    for dialog in test_dialog:
        u_lists = []
        for utts in dialog:
            u_list = []
            for u in utts:
                u = nltk.word_tokenize(u)
                u_list.append(u)
                vocab.index_words(u)
            u_lists.append(u_list)
        data_test['dialog'].append(u_lists)

    for target in test_target:
        u_list = []
        for u in target:
            u = nltk.word_tokenize(u)
            u_list.append(u)
            vocab.index_words(u)
        data_test['target'].append(u_list)

    for situation in test_situation:
        u_list = []
        for u in situation:
            u = nltk.word_tokenize(u)
            u_list.append(u)
            vocab.index_words(u)
        data_test['situation'].append(u_list)

    for emotion in test_emotion:
        data_test['emotion'].append(emotion)

    for cause in test_usercause:
        u_lists = []
        for utts in cause:
            u_list = nltk.word_tokenize(utts)
            u_lists.append(u_list)
        data_test['usercause'].append(u_lists)

    # for cause in test_botcause:
    #     u_lists = []
    #     for utts in cause:
    #         u_list = nltk.word_tokenize(utts)
    #         u_lists.append(u_list)
    #     data_test['botcause'].append(u_lists)

    # for label in test_usercause_label:
    #     data_test['usercause_label'].append(label)

    # for label in test_botcause_label:
    #     data_test['botcause_label'].append(label)

    assert len(data_test['dialog']) == len(data_test['target']) == len(data_test['emotion']) == len(
        data_test['situation']) == len(data_test['usercause']) == len(data_test['usercause_label'])

    T = 2
    # >>>>>>>>>> causes of the conversation train >>>>>>>>>> #
    if not os.path.exists(config.data_npy_dict + 'sys_UserCause_{}_Path.train.json'.format(T)):
        train_usercause = np.load(config.data_npy_dict + 'sys_usercause_texts.train.npy',
                                  allow_pickle=True)
        results, causes, idlist = [], [], []
        for idx, dialog in enumerate(train_dialog):
            for jdx, utts in enumerate(dialog[-1]):
                idlist.append(idx)
                results.append(utts)
                causes.append(train_usercause[idx][jdx])

        newIdlist = process(config.data_npy_dict + 'sys_UserCause_{}_Path.train.json'.format(T), causes, results, idlist, T)
        with open(config.data_npy_dict + 'sys_UserCause_{}_PathId.train.json'.format(T), 'w') as f:
            json.dump(newIdlist, f)

    if not os.path.exists(config.data_npy_dict + 'sys_UserCause_{}_Path.valid.json'.format(T)):
        dev_usercause = np.load(config.data_npy_dict + 'sys_usercause_texts.valid.npy',
                                allow_pickle=True)
        results, causes, idlist = [], [], []
        for idx, dialog in enumerate(dev_dialog):
            for jdx, utts in enumerate(dialog[-1]):
                idlist.append(idx)
                results.append(utts)
                causes.append(dev_usercause[idx][jdx])

        newIdlist = process(config.data_npy_dict + 'sys_UserCause_{}_Path.valid.json'.format(T), causes, results, idlist, T)
        with open(config.data_npy_dict + 'sys_UserCause_{}_PathId.valid.json'.format(T), 'w') as f:
            json.dump(newIdlist, f)

    if not os.path.exists(config.data_npy_dict + 'sys_UserCause_{}_Path.test.json'.format(T)):
        test_usercause = np.load(config.data_npy_dict + 'sys_usercause_texts.test.npy',
                                 allow_pickle=True)
        results, causes, idlist = [], [], []
        for idx, dialog in enumerate(test_dialog):
            for jdx, utts in enumerate(dialog[-1]):
                idlist.append(idx)
                results.append(utts)
                causes.append(test_usercause[idx][jdx])

        newIdlist = process(config.data_npy_dict + 'sys_UserCause_{}_Path.test.json'.format(T), causes, results, idlist, T)
        with open(config.data_npy_dict + 'sys_UserCause_{}_PathId.test.json'.format(T), 'w') as f:
            json.dump(newIdlist, f)

    if not os.path.exists(config.data_concept_dict + 'sys_UserHops_{}_triple.train.json'.format(T)):
        usergraph = directed_triple(config.data_npy_dict + 'sys_UserCause_{}_Path.train.json'.format(T),
                        config.data_concept_dict + 'sys_UserHops_{}_triple.train.json'.format(T))
    else:
        usergraph = read_json(config.data_concept_dict + 'sys_UserHops_{}_triple.train.json'.format(T))
    with open(config.data_npy_dict + 'sys_UserCause_{}_PathId.train.json'.format(T), 'r') as f:
        data_train['graphidx'] = json.load(f)

    data_train['graphs'] = usergraph

    assert len(data_train['target']) == len(data_train['graphidx'])

    if not os.path.exists(config.data_concept_dict + 'sys_UserHops_{}_triple.valid.json'.format(T)):
        usergraph = directed_triple(config.data_npy_dict + 'sys_UserCause_{}_Path.valid.json'.format(T),
                        config.data_concept_dict + 'sys_UserHops_{}_triple.valid.json'.format(T))
    else:

        usergraph = read_json(config.data_concept_dict + 'sys_UserHops_{}_triple.valid.json'.format(T))
    with open(config.data_npy_dict + 'sys_UserCause_{}_PathId.valid.json'.format(T), 'r') as f:
        data_dev['graphidx'] = json.load(f)

    data_dev['graphs'] = usergraph

    assert len(data_dev['target']) == len(data_dev['graphidx'])

    if not os.path.exists(config.data_concept_dict + 'sys_UserHops_{}_triple.test.json'.format(T)):
        usergraph = directed_triple(config.data_npy_dict + 'sys_UserCause_{}_Path.test.json'.format(T),
                        config.data_concept_dict + 'sys_UserHops_{}_triple.test.json'.format(T))
    else:
        usergraph = read_json(config.data_concept_dict + 'sys_UserHops_{}_triple.test.json'.format(T))
    with open(config.data_npy_dict + 'sys_UserCause_{}_PathId.test.json'.format(T), 'r') as f:
        data_test['graphidx'] = json.load(f)

    data_test['graphs'] = usergraph

    assert len(data_test['target']) == len(data_test['graphidx'])

    return data_train, data_dev, data_test, vocab

def load_dataset():
    if os.path.exists(config.data_path):
        print("LOADING empathetic_dialogue")
        with open(config.data_path, "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
            # >>>>>>>>>> dictionaries >>>>>>>>>> #
    else:
        print("Building dataset...")
        data_tra, data_val, data_tst, vocab = read_langs(vocab=Lang(
            {config.UNK_idx: "UNK", config.PAD_idx: "PAD", config.EOS_idx: "EOS", config.SOS_idx: "SOS",
             config.USR_idx: "USR", config.SYS_idx: "SYS", config.SIT_idx: "SIT", config.CLS_idx: "CLS",
             config.SEP_idx: "SEP"}))
        with open(config.data_path, "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab], f)
            print("Saved PICKLE")
    for i in range(3):
        print('[situation]:', ' '.join([ele for lis in data_tra['situation'][i] for ele in lis]))
        print('[emotion]:', data_tra['emotion'][i])
        print('[dialog]:', [' '.join(u) for u in [ele for lis in data_tra['dialog'][i] for ele in lis]])
        print('[target]:', ' '.join([ele for lis in data_tra['target'][i] for ele in lis]))
        print(" ")
    return data_tra, data_val, data_tst, vocab

