import numpy as np
import copy

from PCFG_tree import *
from oov import NormalizeSentence

def GetPossibleParsings(sentence, grammar, inv_grammar, lexicon):
    C = [[]]
    n = len(sentence)
    for word in sentence:
        trees = set()
        for pos in lexicon[word].keys():
            par = Node(None, pos)
            child = Node(par, word)
            trees = trees.union({PCFG_Tree(root=par)})
        C[0].append(trees)
    
    for i in range(1, n):
        C.append([])
        for j in range(n-i):
            trees = set()
            for k in range(i):
                for tree1 in C[k][j]:
                    for tree2 in C[i-k-1][j+k+1]:
                        ch = tree1.root.value + ' ' + tree2.root.value
                        if ch in inv_grammar:
                            for tag in inv_grammar[ch]:
                                tree1_c = copy.deepcopy(tree1)
                                tree2_c = copy.deepcopy(tree2)
                                root = Node(None, tag)
                                tree1_c.root.parent = root
                                tree2_c.root.parent = root
                                root.children = [tree1_c.root, tree2_c.root]
                                trees = trees.union({PCFG_Tree(root=root)})  
            C[i].append(trees)
    trees = list(filter(lambda x: x.root.value == 'SENT', C[-1][0]))
    return trees

def ParseSentence(sentence, embeddings, word_id_big,\
              id_word_big, word_id, grammar, inv_grammar, lexicon, mat):

    norm_sentence = NormalizeSentence(sentence, embeddings, word_id_big, id_word_big, word_id, mat)
    trees = GetPossibleParsings(norm_sentence, grammar, inv_grammar, lexicon)
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