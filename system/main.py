import os
import argparse

from preprocess import * 
from load_models import *
from cyk import ParseSentence

# Read arguments
p = argparse.ArgumentParser( description='Argument parser' )

p.add_argument( '--file', type=str, required=True, help='Input file path' )
args = p.parse_args()

if not os.path.exists('data/train_in.txt'):
    print('Preprocessing data')
    Preprocess()
    print('Preprocess done')

print('Loading models...')
word_id, id_word = LoadDictionary()
Bigram_Matrix = LoadLanguageModel(word_id)
Grammar, Lexicon, Inv_Grammar = LoadGrammarAndLexicon()
embeddings, word_id_big, id_word_big = LoadEmbeddings()
print('Done')

filename = args.file.split('/')[-1].replace('.txt', '_out.txt')

if not os.path.exists('output'):
	os.mkdir('output')

file = open('output/' + filename, 'w')

data = open(args.file, 'r').readlines()
n = str(len(data))
i = 1
for sent in data:
	print('Parsing sentence ' + str(i) + '/' + n + '')
	i += 1
	parsed = ParseSentence(sent, embeddings, word_id_big,\
	             id_word_big, word_id, Grammar, Inv_Grammar, Lexicon, Bigram_Matrix)
	file.write(parsed + '\n')

file.close()
print('Parisng finished')

    