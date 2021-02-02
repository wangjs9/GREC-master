import pandas as pd
import numpy as np
import nltk
# from nltk.tokenize import WordPunctTokenizer

sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

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



