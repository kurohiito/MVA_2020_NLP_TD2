import numpy as np
import copy
import time

from PCFG_tree import *
from oov import NormalizeSentence


def GetPossibleParsings(sentence, grammar, inv_grammar, lexicon, time_lim):
    C = [[]]
    n = len(sentence)
    for word in sentence:
        trees = set()
        for pos in lexicon[word].keys():
            par = Node(None, pos)
            child = Node(par, word)
            trees = trees.union({par})
        C[0].append(trees)
    start = time.time()
    for i in range(1, n):
        C.append([])
        for j in range(n-i):
            trees = []
            for k in range(i):
                first_tags = list(filter(lambda x: not '|' in x.value, C[k][j]))
                if time.time() - start > time_lim:
                    return []
                for tree1 in first_tags:
                    for tree2 in C[i-k-1][j+k+1]:
                        ch = tree1.value + ' ' + tree2.value
                        if ch in inv_grammar:             
                            for tag in inv_grammar[ch]:
                                cond1 = '|' not in tag
                                if '|' in tag:
                                    l = len(tag.split('|'))
                                    cond2 = j + 1 >= l
                                if (cond1 or cond2):
                                    root = Node(None, tag)
                                    root.children = [tree1, tree2]
                                    tag_list = list(map(lambda x: x.value, trees))
                                    if tag in tag_list:
                                        idx = tag_list.index(tag)
                                        p1 = grammar[tag][' '.join(trees[idx].GetChildrenValues())]
                                        p2 = grammar[tag][ch]
                                        if p2 > p1:
                                            trees.append(root)
                                    else:
                                        trees.append(root)
            C[i].append(set(trees))
    trees = []
    for tree in C[-1][0]:
        if 'SENT' in tree.value:
            trees.append(PCFG_Tree(root=tree))
    return trees
                
def ParseSentence(sentence, embeddings, word_id_big,\
              id_word_big, word_id, grammar, inv_grammar, lexicon, mat, time_lim):

    norm_sentence = NormalizeSentence(sentence, embeddings, word_id_big, id_word_big, word_id, mat)
    trees = GetPossibleParsings(norm_sentence, grammar, inv_grammar, lexicon, time_lim)
    if len(trees) == 0:
        return "Cannot find valid parsing"

    probas = list(map(lambda x: x.GetProba(lexicon, grammar), trees))
    
    best_parsing = trees[np.argmax(probas)]
    best_parsing.InvTransformFromCNF()
    parsed_sentence = best_parsing.GetSentence()
    
    sentence_split = sentence.split()
    for i in range(len(norm_sentence)):
        if sentence_split[i] != norm_sentence[i]:
            parsed_sentence = parsed_sentence.replace(norm_sentence[i], sentence_split[i])
    return(parsed_sentence)
