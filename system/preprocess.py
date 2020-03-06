import numpy as np
import re

def Preprocess():
    raw_data = open('data/sequoia-corpus+fct.mrg_strict', 'r').readlines()
    n = len(raw_data)
    idx = np.arange(n)

    train_idx = idx[:int(0.8*n)]
    val_idx = idx[int(0.8*n):int(0.9*n)]

    train_in = open('data/train_in.txt', 'w')
    test_in = open('data/test_in.txt', 'w')
    dev_in = open('data/dev_in.txt', 'w')
    train_out = open('data/train_out.txt', 'w')
    test_out = open('data/test_out.txt', 'w')
    dev_out = open('data/dev_out.txt', 'w')

    for i in range(n):
        raw_data[i] = re.sub(r'-[A-Z]{3}|-[A-Z]{1,}_[A-Z]{1,}', '', raw_data[i])
        mots = list(filter(lambda s: s[-1] == ')', raw_data[i].split()))
        mots = list(map(lambda s: s.replace(')', ''), mots))
        sentence = ' '.join(mots)
        
        if i in train_idx:
            train_in.write(sentence + '\n')
            train_out.write(raw_data[i])
        elif i in val_idx:
            test_in.write(sentence + '\n')
            test_out.write(raw_data[i])
        else:
            dev_in.write(sentence + '\n')
            dev_out.write(raw_data[i])
            
    train_in.close()
    test_in.close()
    dev_in.close()
    train_out.close()
    test_out.close()
    dev_out.close()