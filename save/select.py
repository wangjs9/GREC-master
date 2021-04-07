import numpy as np 
import pandas as pd

folders = ['mime', 'multiexpert-6-8', 'multihop-6-8', 'trs', 'w.o/refer', 'w.o/encoder', 'w.o/graph']

def context(numbers):
    batch = pd.read_csv('batch.csv')
    emotion = pd.read_csv('emotion.csv')
    reply = pd.read_csv('reply_true.csv')
    


def select(folder_name):
    with open('numbers.csv', 'r') as f:
        numbers = f.readlines()
        numbers = [int(n.strip()) for n in numbers]
    context(numbers)
    with open(folder_name+'/reply_beam.csv', 'r') as f:
        replies = f.readlines()
        replies = [r.strip() for r in replies]
        replies = np.array(replies)
        replies = replies[numbers]
    replies.tolist()
    return replies
    
        
def recover():
    pass


context([0])