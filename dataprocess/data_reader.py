import config
import numpy as np
import nltk
import os, pickle, json
from models.cause_extraction import predict
from dataprocess.find_path import process
from dataprocess.triple_store import directed_triple, read_json

blacklist = set(["from", "as", "more", "either", "in", "and", "on", "an", "when", "too", "to", "i", "do", "can", "be", "that", "or", "the", "a", "of", "for", "is", "was", "the", "-PRON-", "actually", "likely", "possibly", "want",
                 "make", "my", "someone", "sometimes_people", "sometimes","would", "want_to", "one", "something", "sometimes", "everybody", "somebody", "could", "could_be","mine","us","em",
                 "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "A", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "after", "afterwards", "ag", "again", "against", "ah", "ain", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appreciate", "approximately", "ar", "are", "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "B", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "been", "before", "beforehand", "beginnings", "behind", "below", "beside", "besides", "best", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "C", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "ci", "cit", "cj", "cl", "clearly", "cm", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "could", "couldn", "couldnt", "course", "cp", "cq", "cr", "cry", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d", "D", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "dj", "dk", "dl", "do", "does", "doesn", "doing", "don", "done", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "E", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "F", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "G", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "H", "h2", "h3", "had", "hadn", "happens", "hardly", "has", "hasn", "hasnt", "have", "haven", "having", "he", "hed", "hello", "help", "hence", "here", "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hh", "hi", "hid", "hither", "hj", "ho", "hopefully", "how", "howbeit", "however", "hr", "hs", "http", "hu", "hundred", "hy", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "im", "immediately", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "inward", "io", "ip", "iq", "ir", "is", "isn", "it", "itd", "its", "iv", "ix", "iy", "iz", "j", "J", "jj", "jr", "js", "jt", "ju", "just", "k", "K", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "ko", "l", "L", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "M", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "my", "n", "N", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "neither", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "O", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "otherwise", "ou", "ought", "our", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "P", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "pp", "pq", "pr", "predominantly", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "Q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "R", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "S", "s2", "sa", "said", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "seem", "seemed", "seeming", "seems", "seen", "sent", "seven", "several", "sf", "shall", "shan", "shed", "shes", "show", "showed", "shown", "showns", "shows", "si", "side", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somehow", "somethan", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "sz", "t", "T", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "thereof", "therere", "theres", "thereto", "thereupon", "these", "they", "theyd", "theyre", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "U", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "used", "useful", "usefully", "usefulness", "using", "usually", "ut", "v", "V", "va", "various", "vd", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "W", "wa", "was", "wasn", "wasnt", "way", "we", "wed", "welcome", "well", "well-b", "went", "were", "weren", "werent", "what", "whatever", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "whom", "whomever", "whos", "whose", "why", "wi", "widely", "with", "within", "without", "wo", "won", "wonder", "wont", "would", "wouldn", "wouldnt", "www", "x", "X", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "Y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "your", "youre", "yours", "yr", "ys", "yt", "z", "Z", "zero", "zi", "zz"])

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
    ScaleType = 'sys'

    # >>>>>>>>>> historical utterances >>>>>>>>>> #
    train_dialog = np.load(config.data_npy_dict+'{}_dialog_texts.train.npy'.format(ScaleType), allow_pickle=True)
    # >>>>>>>>>> next expected utterance from the bot >>>>>>>>>> #
    train_target = np.load(config.data_npy_dict+'{}_target_texts.train.npy'.format(ScaleType), allow_pickle=True)
    # >>>>>>>>>> emotions of the conversation >>>>>>>>>> #
    train_emotion = np.load(config.data_npy_dict+'{}_emotion_texts.train.npy'.format(ScaleType), allow_pickle=True)
    # >>>>>>>>>> prompts of the conversation >>>>>>>>>> #
    train_situation = np.load(config.data_npy_dict+'{}_situation_texts.train.npy'.format(ScaleType), allow_pickle=True)
    necessary_file = ['usercause_texts', 'botcause_texts', 'usercause_labels', 'botcause_labels']
    for file in necessary_file:
        if not os.path.exists(config.data_npy_dict + '{}_{}.train.npy'.format(ScaleType, file)):
            print('Generating cause clauses...')
            predict(train_situation, train_dialog, train_target, 'train', config.data_npy_dict)
            break
    # >>>>>>>>>> usercause of the conversation >>>>>>>>>> #
    train_usercause = np.load(config.data_npy_dict + '{}_usercause_texts.train.npy'.format(ScaleType), allow_pickle=True)
    # >>>>>>>>>> botcause of the conversation >>>>>>>>>> #
    train_botcause = np.load(config.data_npy_dict + '{}_botcause_texts.train.npy'.format(ScaleType), allow_pickle=True)
    # >>>>>>>>>> usercause label of the conversation >>>>>>>>>> #
    train_usercause_label = np.load(config.data_npy_dict + '{}_usercause_labels.train.npy'.format(ScaleType),
                              allow_pickle=True)
    # >>>>>>>>>> botcause label of the conversation >>>>>>>>>> #
    train_botcause_label = np.load(config.data_npy_dict + '{}_botcause_labels.train.npy'.format(ScaleType),
                              allow_pickle=True)

    dev_dialog = np.load(config.data_npy_dict+'{}_dialog_texts.valid.npy'.format(ScaleType), allow_pickle=True)
    dev_target = np.load(config.data_npy_dict+'{}_target_texts.valid.npy'.format(ScaleType), allow_pickle=True)
    dev_emotion = np.load(config.data_npy_dict+'{}_emotion_texts.valid.npy'.format(ScaleType), allow_pickle=True)
    dev_situation = np.load(config.data_npy_dict+'{}_situation_texts.valid.npy'.format(ScaleType), allow_pickle=True)
    for file in necessary_file:
        if not os.path.exists(config.data_npy_dict + '{}_{}.valid.npy'.format(ScaleType, file)):
            print('Generating cause clauses...')
            predict(dev_situation, dev_dialog, dev_target, 'valid', config.data_npy_dict)
    dev_usercause = np.load(config.data_npy_dict + '{}_usercause_texts.valid.npy'.format(ScaleType), allow_pickle=True)
    dev_botcause = np.load(config.data_npy_dict + '{}_botcause_texts.valid.npy'.format(ScaleType), allow_pickle=True)
    dev_usercause_label = np.load(config.data_npy_dict + '{}_usercause_labels.valid.npy'.format(ScaleType),
                                    allow_pickle=True)
    dev_botcause_label = np.load(config.data_npy_dict + '{}_botcause_labels.valid.npy'.format(ScaleType),
                                   allow_pickle=True)

    test_dialog = np.load(config.data_npy_dict+'{}_dialog_texts.test.npy'.format(ScaleType), allow_pickle=True)
    test_target = np.load(config.data_npy_dict+'{}_target_texts.test.npy'.format(ScaleType), allow_pickle=True)
    test_emotion = np.load(config.data_npy_dict+'{}_emotion_texts.test.npy'.format(ScaleType), allow_pickle=True)
    test_situation = np.load(config.data_npy_dict+'{}_situation_texts.test.npy'.format(ScaleType), allow_pickle=True)
    for file in necessary_file:
        if not os.path.exists(config.data_npy_dict + '{}_{}.test.npy'.format(ScaleType, file)):
            print('Generating cause clauses...')
            predict(test_situation, test_dialog, test_target, 'test', config.data_npy_dict)
    test_usercause = np.load(config.data_npy_dict + '{}_usercause_texts.test.npy'.format(ScaleType), allow_pickle=True)
    test_botcause = np.load(config.data_npy_dict + '{}_botcause_texts.test.npy'.format(ScaleType), allow_pickle=True)
    test_usercause_label = np.load(config.data_npy_dict + '{}_usercause_labels.test.npy'.format(ScaleType),
                                  allow_pickle=True)
    test_botcause_label = np.load(config.data_npy_dict + '{}_botcause_labels.test.npy'.format(ScaleType),
                                 allow_pickle=True)

    data_train = {'dialog': [], 'target': [], 'emotion': [], 'situation': [], 'usercuase_label': [], 'botcause_label': [], 'usercause':[], 'botcause': [], 'graph':[], 'graphidx': []}
    data_dev = {'dialog': [], 'target': [], 'emotion': [], 'situation': [], 'usercuase_label': [], 'botcause_label': [], 'usercause': [], 'botcause': [], 'graph': [], 'graphidx': []}
    data_test = {'dialog': [], 'target': [], 'emotion': [], 'situation': [], 'usercuase_label': [], 'botcause_label': [], 'usercause': [], 'botcause': [], 'graph': [], 'graphidx': []}

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

    for emotion in train_emotion:
        data_train['emotion'].append(emotion)

    for cause in train_usercause:
        data_train['usercause'].append(cause)

    for cause in train_botcause:
        data_train['botcause'].append(cause)

    for label in train_usercause_label:
        data_train['usercuase_label'].append(label)

    for label in train_botcause_label:
        data_train['botcause_label'].append(label)

    assert len(data_train['dialog']) == len(data_train['target']) == len(data_train['emotion']) == len(
        data_train['situation']) == len(data_train['usercause']) == len(data_train['botcause']) == len(
        data_train['usercuase_label']) == len(data_train['botcause_label'])

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
        data_dev['usercause'].append(cause)

    for cause in dev_botcause:
        data_dev['botcause'].append(cause)

    for label in dev_usercause_label:
        data_dev['usercuase_label'].append(label)

    for label in dev_botcause_label:
        data_dev['botcause_label'].append(label)

    assert len(data_dev['dialog']) == len(data_dev['target']) == len(data_dev['emotion']) == len(
        data_dev['situation']) == len(data_dev['usercause']) == len(data_dev['botcause']) == len(
        data_dev['usercuase_label']) == len(data_dev['botcause_label'])

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
        data_test['usercause'].append(cause)

    for cause in test_botcause:
        data_test['botcause'].append(cause)

    for label in test_usercause_label:
        data_test['usercuase_label'].append(label)

    for label in test_botcause_label:
        data_test['botcause_label'].append(label)

    assert len(data_test['dialog']) == len(data_test['target']) == len(data_test['emotion']) == len(
        data_test['situation']) == len(data_test['usercause']) == len(data_test['botcause']) == len(
        data_test['usercuase_label']) == len(data_test['botcause_label'])

    T = 2
    # >>>>>>>>>> causes of the conversation train >>>>>>>>>> #
    if not os.path.exists(config.data_npy_dict + '{}_UserCause_{}_Path.train.json'.format(ScaleType, T)):
        train_usercause = np.load(config.data_npy_dict + '{}_usercause_texts.train.npy'.format(ScaleType),
                                  allow_pickle=True)
        results, causes, idlist = [], [], []
        for idx, dialog in enumerate(train_dialog):
            for jdx, utts in enumerate(dialog[-1]):
                idlist.append(idx)
                results.append(utts)
                causes.append(train_usercause[idx][jdx])

        newIdlist = process(config.data_npy_dict + '{}_UserCause_{}_Path.train.json'.format(ScaleType, T), causes, results, idlist, T)
        with open(config.data_npy_dict + '{}_UserCause_{}_PathId.train.json'.format(ScaleType, T), 'w') as f:
            json.dump(newIdlist, f)

    # if not os.path.exists(config.data_npy_dict + '{}_BotCause_{}_Path.train.json'.format(ScaleType, T)):
    #     train_botcause = np.load(config.data_npy_dict + '{}_botcause_texts.train.npy'.format(ScaleType),
    #                              allow_pickle=True)
    #     results, causes, idlist = [], [], []
    #     for idx, dialog in enumerate(train_target):
    #         for jdx, utts in enumerate(dialog):
    #             idlist.append(idx)
    #             results.append(utts)
    #             causes.append(train_botcause[idx][jdx])
    #
    #     process(config.data_npy_dict + '{}_BotCause_{}_Path.train.json'.format(ScaleType, T), causes, results, T)

    # >>>>>>>>>> causes of the conversation valid >>>>>>>>>> #
    if not os.path.exists(config.data_npy_dict + '{}_UserCause_{}_Path.valid.json'.format(ScaleType, T)):
        dev_usercause = np.load(config.data_npy_dict + '{}_usercause_texts.valid.npy'.format(ScaleType),
                                allow_pickle=True)
        results, causes, idlist = [], [], []
        for idx, dialog in enumerate(dev_dialog):
            for jdx, utts in enumerate(dialog[-1]):
                idlist.append(idx)
                results.append(utts)
                causes.append(dev_usercause[idx][jdx])

        newIdlist = process(config.data_npy_dict + '{}_UserCause_{}_Path.valid.json'.format(ScaleType, T), causes, results, idlist, T)
        with open(config.data_npy_dict + '{}_UserCause_{}_PathId.valid.json'.format(ScaleType, T), 'w') as f:
            json.dump(newIdlist, f)

    # if not os.path.exists(config.data_npy_dict + '{}_BotCause_{}_Path.valid.json'.format(ScaleType, T)):
    #     dev_botcause = np.load(config.data_npy_dict + '{}_botcause_texts.valid.npy'.format(ScaleType),
    #                            allow_pickle=True)
    #     results, causes, idlist = [], [], []
    #     for idx, dialog in enumerate(dev_target):
    #         for jdx, utts in enumerate(dialog):
    #             idlist.append(idx)
    #             results.append(utts)
    #             causes.append(dev_botcause[idx][jdx])
    #
    #     process(config.data_npy_dict + '{}_BotCause_{}_Path.valid.json'.format(ScaleType, T), causes, results, T)

    # >>>>>>>>>> causes of the conversation test >>>>>>>>>> #
    if not os.path.exists(config.data_npy_dict + '{}_UserCause_{}_Path.test.json'.format(ScaleType, T)):
        test_usercause = np.load(config.data_npy_dict + '{}_usercause_texts.test.npy'.format(ScaleType),
                                 allow_pickle=True)
        results, causes, idlist = [], [], []
        for idx, dialog in enumerate(test_dialog):
            for jdx, utts in enumerate(dialog[-1]):
                idlist.append(idx)
                results.append(utts)
                causes.append(test_usercause[idx][jdx])

        newIdlist = process(config.data_npy_dict + '{}_UserCause_{}_Path.test.json'.format(ScaleType, T), causes, results, idlist, T)
        with open(config.data_npy_dict + '{}_UserCause_{}_PathId.test.json'.format(ScaleType, T), 'w') as f:
            json.dump(newIdlist, f)

    # if not os.path.exists(config.data_npy_dict + '{}_BotCause_{}_Path.test.json'.format(ScaleType, T)):
    #     test_botcause = np.load(config.data_npy_dict + '{}_botcause_texts.test.npy'.format(ScaleType),
    #                             allow_pickle=True)
    #
    #     results, causes, idlist = [], [], []
    #     for idx, dialog in enumerate(test_target):
    #         for jdx, utts in enumerate(dialog):
    #             idlist.append(idx)
    #             results.append(utts)
    #             causes.append(test_botcause[idx][jdx])
    #
    #     process(config.data_npy_dict + '{}_BotCause_{}_Path.test.json'.format(ScaleType, T), causes, results, T)

    if not os.path.exists(config.data_concept_dict + '{}_UserHops_{}_triple.train.json'.format(ScaleType, T)):
        usergraph = directed_triple(config.data_npy_dict + '{}_UserCause_{}_Path.train.json'.format(ScaleType, T),
                        config.data_concept_dict + '{}_UserHops_{}_triple.train.json'.format(ScaleType, T))
    else:
        with open(config.data_npy_dict + '{}_UserCause_{}_PathId.train.json'.format(ScaleType, T), 'r') as f:
            data_train['graphidx'] = json.load(f)
        usergraph = read_json(config.data_concept_dict + '{}_UserHops_{}_triple.train.json'.format(ScaleType, T))
    # if not os.path.exists(config.data_concept_dict + '{}_BotHops_{}_triple.train.json'.format(ScaleType, T)):
    #     botgraph = directed_triple(config.data_npy_dict + '{}_BotCause_{}_Path.train.json'.format(ScaleType, T),
    #                     config.data_concept_dict + '{}_BotHops_{}_triple.train.json'.format(ScaleType, T))
    # else:
    #     botgraph = read_json(config.data_concept_dict + '{}_BotHops_{}_triple.train.json'.format(ScaleType, T))

    data_train['graph'] = usergraph
    # data_train['botgraph'] = botgraph

    assert len(data_train['target']) == len(data_train['graphidx'])

    if not os.path.exists(config.data_concept_dict + '{}_UserHops_{}_triple.valid.json'.format(ScaleType, T)):
        usergraph = directed_triple(config.data_npy_dict + '{}_UserCause_{}_Path.valid.json'.format(ScaleType, T),
                        config.data_concept_dict + '{}_UserHops_{}_triple.valid.json'.format(ScaleType, T))
    else:
        with open(config.data_npy_dict + '{}_UserCause_{}_PathId.valid.json'.format(ScaleType, T), 'r') as f:
            data_dev['graphidx'] = json.load(f)
        usergraph = read_json(config.data_concept_dict + '{}_UserHops_{}_triple.valid.json'.format(ScaleType, T))
    # if not os.path.exists(config.data_concept_dict + '{}_BotHops_{}_triple.valid.json'.format(ScaleType, T)):
    #     botgraph = directed_triple(config.data_npy_dict + '{}_BotCause_{}_Path.valid.json'.format(ScaleType, T),
    #                     config.data_concept_dict + '{}_BotHops_{}_triple.valid.json'.format(ScaleType, T))
    # else:
    #     botgraph = read_json(config.data_concept_dict + '{}_BotHops_{}_triple.valid.json'.format(ScaleType, T))

    data_dev['graph'] = usergraph
    # data_dev['botgraph'] = botgraph

    assert len(data_dev['target']) == len(data_dev['graphidx'])

    if not os.path.exists(config.data_concept_dict + '{}_UserHops_{}_triple.test.json'.format(ScaleType, T)):
        usergraph = directed_triple(config.data_npy_dict + '{}_UserCause_{}_Path.test.json'.format(ScaleType, T),
                        config.data_concept_dict + '{}_UserHops_{}_triple.test.json'.format(ScaleType, T))
    else:
        with open(config.data_npy_dict + '{}_UserCause_{}_PathId.test.json'.format(ScaleType, T), 'w') as f:
            data_test['graphidx'] = json.load(f)
        usergraph = read_json(config.data_concept_dict + '{}_UserHops_{}_triple.test.json'.format(ScaleType, T))
    # if not os.path.exists(config.data_concept_dict + '{}_BotHops_{}_triple.test.json'.format(ScaleType, T)):
    #     botgraph = directed_triple(config.data_npy_dict + '{}_BotCause_{}_Path.test.json'.format(ScaleType, T),
    #                     config.data_concept_dict + '{}_BotHops_{}_triple.test.json'.format(ScaleType, T))
    # else:
    #     botgraph = read_json(config.data_concept_dict + '{}_BotHops_{}_triple.test.json'.format(ScaleType, T))

    data_test['graph'] = usergraph
    # data_test['botgraph'] = botgraph

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
        print('[dialog]:', [' '.join(u) for u in [ele for lis in data_tra['context'][i] for ele in lis]])
        print('[target]:', ' '.join(data_tra['target'][i]))
        print(" ")
    return data_tra, data_val, data_tst, vocab

