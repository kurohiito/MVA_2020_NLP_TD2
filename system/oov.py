import numpy as np
import re

#Change digits with #-s
DIGITS = re.compile("[0-9]", re.UNICODE)

def LevDistance(s1, s2):
    l1 = list(s1)
    l2 = list(s2)
    n1 = len(l1)
    n2 = len(l2)
    m = np.zeros((n1+1, n2+1))
    m[:, 0] = np.arange(n1+1)
    m[0, :] = np.arange(n2+1)
    for i in range(1, n1+1):
        for j in range(1, n2+1):
            if l1[i-1] == l2[j-1]:
                m[i, j] = min(m[i-1, j]+1, m[i, j-1]+1, m[i-1, j-1])
            else:
                m[i, j] = min(m[i-1, j]+1, m[i, j-1]+1, m[i-1, j-1]+1)
    return m[-1, -1]

def GetClosestWords(new_word, dictionary, k):
    closestWords = []
    new_word = DIGITS.sub("#", new_word)
    for word in dictionary:
        if LevDistance(word, new_word) <= k:
            closestWords.append(word)
    
    return(closestWords)

def CaseNormalizer(word, dictionary):
    w = word
    lower = (dictionary.get(w.lower(), 1e12), w.lower())
    upper = (dictionary.get(w.upper(), 1e12), w.upper())
    title = (dictionary.get(w.title(), 1e12), w.title())
    results = [lower, upper, title]
    results.sort()
    index, w = results[0]
    if index != 1e12:
        return w
    return word


def Normalize(word, word_id):
    """ Find the closest alternative in case the word is OOV."""
    if not word in word_id:
        word = DIGITS.sub("#", word)
    if not word in word_id:
        word = CaseNormalizer(word, word_id)

    if not word in word_id:
        return None
    return word

def GetClosestWordsByEmbeddings(word, embeddings, word_id_big, id_word_big, word_id):
    word = Normalize(word, word_id_big)
    if word == None:
        return []
    
    e = embeddings[word_id_big[word]]
    distances = (((embeddings - e) ** 2).sum(axis=1) ** 0.5)
    idx = np.argsort(distances)
    words = []
    for i in idx:
        if id_word_big[i] in word_id:
            words.append(id_word_big[i])
        if len(words) == 5:
            break
    return words

def GetCandidatesWords(word, embeddings, word_id_big, id_word_big, word_id):
    k = 1
    candidates = GetClosestWordsByEmbeddings(word, embeddings, word_id_big, id_word_big, word_id)
    candidates = candidates + GetClosestWords(word, word_id, k)
    
    while len(candidates) == 0:
        k += 1
        candidates = GetClosestWords(word, word_id, k) 
    return candidates

def GetProba(word1, word2, word3, mat, word_id):
    id1 = word_id[word1]
    id2 = word_id[word2]
    if word3 in word_id:
        id3 = word_id[word3]
        p = mat[id3, id2]
    else: 
        p = 1
    
    return mat[id2, id1]*p

def NormalizeSentence(sentence, embeddings, word_id_big, id_word_big, word_id, mat):
    mots = ['<S>'] + sentence.split() + ['<\S>']
    length = len(mots)
    for i in range(1, length-1):
        if not mots[i] in word_id:
            candidates = GetCandidatesWords(mots[i], embeddings, word_id_big, id_word_big, word_id)
            probas = np.zeros(len(candidates))
            for j in range(len(candidates)):
                probas[j] = GetProba(mots[i-1], candidates[j], mots[i+1], mat, word_id)
            mots[i] = candidates[np.argmax(probas)]
    
    return(mots[1:-1])