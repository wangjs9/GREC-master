import pandas as pd
import numpy as np
import nltk
from tqdm import tqdm
import random
# from nltk.tokenize import WordPunctTokenizer
sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
from collections import defaultdict

import config

def clean(sentence, word_pairs):
    sentence = sentence.replace('_comma_', ',')
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    sentence = sen_tokenizer.tokenize(sentence)
    return sentence

def ToClause(DataType):
    if DataType not in ['train', 'valid', 'test']:
        raise ValueError('`DataType` must be in [`train`, `valid`, `test`]')
    print('Loading data...')
    conversation = pd.read_csv(config.data_dict +'{}.csv'.format(DataType), encoding='unicode_escape', header=0, index_col=False)
    situations = list()
    emotions = list()
    history = list()
    targets = list()

    word_pairs = {"it's": "it is", "don't": "do not", "doesn't": "does not", "didn't": "did not", "you'd": "you would",
                  "you're": "you are", "you'll": "you will", "i'm": "i am", "they're": "they are", "that's": "that is",
                  "what's": "what is", "couldn't": "could not", "i've": "i have", "we've": "we have", "can't": "cannot",
                  "i'd": "i would", "i'd": "i would", "aren't": "are not", "isn't": "is not", "wasn't": "was not",
                  "weren't": "were not", "won't": "will not", "there's": "there is", "there're": "there are"}

    context = []
    for idx, row in conversation.iterrows():
        conId, sentId, label, situ, _, utterance, _ = row
        sentId = int(sentId)

        if sentId == 1:
            context = []
        if sentId % 2 == 1:
            context.append(clean(utterance, word_pairs))
        else:
            emotions.append(label)
            situations.append(clean(situ, word_pairs))
            history.append(context.copy())
            bot = clean(utterance, word_pairs)
            targets.append(bot)
            context.append(bot)

        assert len(emotions) == len(situations) == len(history) == len(targets)

    ScaleTypes = ['min', 'sys']
    print('Saving data for sys...')
    np.save(config.data_npy_dict+'sys_situation_texts.{}.npy'.format(DataType), situations)
    np.save(config.data_npy_dict+'sys_dialog_texts.{}.npy'.format(DataType), history)
    np.save(config.data_npy_dict+'sys_target_texts.{}.npy'.format(DataType), targets)
    np.save(config.data_npy_dict+'sys_emotion_texts.{}.npy'.format(DataType), emotions)

    print('Saving data for min...')
    if DataType == 'train':
        num = 200
    else:
        num = 20
    np.save(config.data_npy_dict+'{}_situation_texts.{}.npy'.format(ScaleTypes[0], DataType), situations[:num])
    np.save(config.data_npy_dict+'{}_dialog_texts.{}.npy'.format(ScaleTypes[0], DataType), history[:num])
    np.save(config.data_npy_dict+'{}_target_texts.{}.npy'.format(ScaleTypes[0], DataType), targets[:num])
    np.save(config.data_npy_dict+'{}_emotion_texts.{}.npy'.format(ScaleTypes[0], DataType), emotions[:num])

def add_act():
    from dataprocess.annotate import predict_emotion, labels
    train_target = np.load(config.data_npy_dict + 'sys_target_texts.train.npy', allow_pickle=True)
    dev_target = np.load(config.data_npy_dict + 'sys_target_texts.valid.npy', allow_pickle=True)
    test_target = np.load(config.data_npy_dict + 'sys_target_texts.test.npy', allow_pickle=True)

    with open(config.data_npy_dict + 'sys_act_label.train', 'r') as f:
        train_original_act = f.readlines()
        train_original_act = [act.strip() for act in train_original_act]

    with open(config.data_npy_dict + 'sys_act_label.valid', 'r') as f:
        dev_original_act = f.readlines()
        dev_original_act = [act.strip() for act in train_original_act]

    with open(config.data_npy_dict + 'sys_act_label.test', 'r') as f:
        test_original_act = f.readlines()
        test_original_act = [act.strip() for act in train_original_act]

    train_acts = []
    for idx, target in tqdm(enumerate(train_target), total=len(train_target)):
        acts = []
        if len(target) == 1:
            acts.append(train_original_act[idx])
        else:
            for t in target:
                predictions = predict_emotion([t])
                predictions = np.array(predictions) * np.array([0] * 32 + [1] * 8 + [0])
                indices = predictions.argsort()[-1:][::-1]
                act = labels[indices[0]]
                acts.append(act)
        train_acts.append(acts)
    np.save(config.data_npy_dict + 'sys_act_texts.train.npy', train_acts)

    dev_acts = []
    for idx, target in tqdm(enumerate(dev_target), total=len(dev_target)):
        acts = []
        if len(target) == 1:
            acts.append(dev_original_act[idx])
        else:
            for t in target:
                predictions = predict_emotion([t])
                predictions = np.array(predictions) * np.array([0] * 32 + [1] * 8 + [0])
                indices = predictions.argsort()[-1:][::-1]
                act = labels[indices[0]]
                acts.append(act)
        dev_acts.append(acts)
    np.save(config.data_npy_dict + 'sys_act_texts.valid.npy', dev_acts)

    test_acts = []
    for idx, target in tqdm(enumerate(test_target), total=len(test_target)):
        acts = []
        if len(target) == 1:
            acts.append(test_original_act[idx])
        else:
            for t in target:
                predictions = predict_emotion([t])
                predictions = np.array(predictions) * np.array([0] * 32 + [1] * 8 + [0])
                indices = predictions.argsort()[-1:][::-1]
                act = labels[indices[0]]
                acts.append(act)
        test_acts.append(acts)
    np.save(config.data_npy_dict + 'sys_act_texts.test.npy', test_acts)

def sample_fake(ty="train"):
    emotions = np.load(config.data_npy_dict + 'sys_emotion_texts.{}.npy'.format(ty), allow_pickle=True)
    acts = np.load(config.data_npy_dict + 'sys_act_texts.{}.npy'.format(ty), allow_pickle=True)
    target = np.load(config.data_npy_dict + 'sys_target_texts.{}.npy'.format(ty), allow_pickle=True)
    positive = ['confident', 'joyful', 'grateful', 'impressed', 'proud', 'excited', 'trusting', 'hopeful', 'faithful', 'prepared', 'content', 'surprised', 'caring']
    negative = ['afraid', 'angry', 'annoyed', 'anticipating', 'anxious', 'apprehensive', 'ashamed', 'devastated', 'disappointed', 'disgusted', 'embarrassed',
                'furious', 'guilty', 'jealous', 'lonely', 'nostalgic', 'sad', 'sentimental', 'terrified']
    fake_same_act = []
    fake_same_emo = []
    emo_map, act_map = defaultdict(list), defaultdict(list)
    positive_emo, negative_emo = [], []
    pair = [(a, e) for a, e in zip(acts, emotions)]
    for idx, p in enumerate(pair):
        a, e = p
        emo_map[e].append(idx)
        for l in a:
            act_map[l].append(idx)
        if e in positive:
            positive_emo.append(idx)
        elif e in negative:
            negative_emo.append(idx)
    positive_emo = set(positive_emo)
    negative_emo = set(negative_emo)
    for idx, p in tqdm(enumerate(pair), total=len(pair)):
        a, e = p
        same_act = []
        for l in a:
            same_act += act_map[l]
        true_same_act = set(act_map[a[0]]+act_map[a[-1]])
        same_act = set(same_act)
        # for same emotion
        same_emotion = set(emo_map[e])
        fake_same_emo.append(random.sample(same_emotion-same_act, 10))
        # for same act
        if e in positive:
            fake_same_act.append(random.sample(true_same_act - positive_emo, 10))
        elif e in negative:
            fake_same_act.append(random.sample(true_same_act - negative_emo, 10))
    np.save(config.data_npy_dict + 'sys_fake_same_act.{}.npy'.format(ty), fake_same_act)
    np.save(config.data_npy_dict + 'sys_fake_same_emo.{}.npy'.format(ty), fake_same_emo)

# add_act()
sample_fake("train")
# sample_fake("valid")
# sample_fake("test")

