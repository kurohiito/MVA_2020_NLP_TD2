import numpy as np
import re
import pickle

from PCFG_tree import PCFG_Tree


#read data
data_in = open('data/train_in.txt', 'r').readlines()
data_out = open('data/train_out.txt', 'r').readlines()

#Change digits with #-s
DIGITS = re.compile("[0-9]", re.UNICODE)

def LoadDictionary():
    #create list of words

    words = []
    for sentence in data_in:
        sentence = DIGITS.sub("#", sentence)
        words.extend(sentence.split())
    words = set(words)
    words = words.union({'<S>', '<\S>'})
    n_words = len(words)

    #create dictionary
    words = list(zip(words, range(n_words)))
    id_word = {i:w for w,i in words}
    word_id = dict(words)
    return word_id, id_word

def LoadLanguageModel(word_id):
	#compute bigram matrix
	n_words = len(word_id)
	Bigram_Matrix = np.ones((n_words, n_words))

	for sentence in data_in:
	    mots = ['<S>'] + sentence.split() + ['<\S>']
	    mots = list(map(lambda x: DIGITS.sub("#", x), mots))
	    n = len(mots)
	    for i in range(0,  n-1):
	        i_id = word_id[mots[i-1]]
	        j_id = word_id[mots[i]]
	        Bigram_Matrix[i_id, j_id] += 1
	for j in range(n_words):
	    Bigram_Matrix[:, j] = Bigram_Matrix[:, j] / Bigram_Matrix[:, j].sum()
	return Bigram_Matrix

def LoadGrammarAndLexicon():
	Grammar = {}
	Lexicon = {}
	Inv_Grammar = {}

	for sentence in data_out:
	    tree = PCFG_Tree(sentence)
	    tree.TransformToCNF()
	    tree.ExtractGrammar()
	    for parent, child in tree.grammar:
	        if parent in Grammar:
	            if child in Grammar[parent]:
	                Grammar[parent][child] += 1
	            else:
	                Grammar[parent][child] = 1
	        else:
	            Grammar[parent] = {child : 1}
	        if child in Inv_Grammar:
	            Inv_Grammar[child] = Inv_Grammar[child].union({parent})
	        else:
	            Inv_Grammar[child] = {parent}
	    for token, pos in tree.lexicon:
	        if token in Lexicon:
	            if pos in Lexicon[token]:
	                Lexicon[token][pos] += 1
	            else:
	                Lexicon[token][pos] = 1
	        else:
	            Lexicon[token] = {pos : 1}

	for par in Grammar:
	    s = sum(Grammar[par].values())
	    for ch in Grammar[par]:
	        Grammar[par][ch] /= s
	for par in Lexicon:
	    s = sum(Lexicon[par].values())
	    for ch in Lexicon[par]:
	        Lexicon[par][ch] /= s
	return Grammar, Lexicon, Inv_Grammar

def LoadEmbeddings():
    words, embeddings = pickle.load(open('data/polyglot-fr.pkl', 'rb'), encoding='latin1')
    word_id_big = {w:i for (i, w) in enumerate(words)}
    id_word_big = dict(enumerate(words))
    return embeddings, word_id_big, id_word_big