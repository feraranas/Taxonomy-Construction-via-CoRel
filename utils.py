from collections import defaultdict
import numpy as np
import os
import torch
from tqdm import tqdm

def load_seed(dataset, file):
    topic_words = {}
    with open(dataset+'/result_'+file+'.txt') as f:
        data=f.readlines()
        current_topic = ''
        for line in data:
            if len(line.strip()) == 0:
                current_topic = ''
                continue
            elif len(line.split(' ')) == 1:
                current_topic = line.split(':')[0]
                continue
            elif current_topic != '':
                topic_words[current_topic] = line.strip().split(' ')

    return topic_words

def get_emb(vec_file):
    f = open(vec_file, 'r')
    contents = f.readlines()[1:]
    word_emb = {}
    vocabulary = {}
    vocabulary_inv = {}
    emb_mat = []
    for i, content in enumerate(contents):
        content = content.strip()
        tokens = content.split(' ')
        word = tokens[0]
        vec = tokens[1:]
        vec = [float(ele) for ele in vec]
        word_emb[word] = np.array(vec)
        vocabulary[word] = i
        vocabulary_inv[i] = word
        emb_mat.append(np.array(vec))
    vocab_size = len(vocabulary)
    emb_mat = np.array(emb_mat) 
    return word_emb, vocabulary, vocabulary_inv, emb_mat

def get_temb(vec_file, topic_file):
    topic2id = {}
    topic_emb = {}
    id2topic = {}
    topic_hier = {}
    i = 0
    with open(topic_file, 'r') as f:
        for line in f:
            parent = line.strip().split('\t')[0]
            temp = line.strip().split('\t')[1]          
            for topic in temp.split(' '):
                topic2id[topic] = i
                id2topic[i] = topic
                i += 1
                if parent not in topic_hier:
                    topic_hier[parent] = []
                topic_hier[parent].append(topic)
    f = open(vec_file, 'r')
    contents = f.readlines()[1:]
    for i, content in enumerate(contents):
        content = content.strip()
        tokens = content.split(' ')
        vec = tokens
        vec = [float(ele) for ele in vec]
        topic_emb[id2topic[i]] = np.array(vec)
    return topic_emb, topic2id, id2topic, topic_hier

def get_cap(vec_file, cap0_file=None):
    f = open(vec_file, 'r')
    contents = f.readlines()[1:]
    word_cap = {}
    for i, content in enumerate(contents):
        content = content.strip()
        tokens = content.split(' ')
        word = tokens[0]
        vec = tokens[1]
        vec = float(vec)
        word_cap[word] = vec
    
    if cap0_file is not None:
        with open(cap0_file) as f:
            contents = f.readlines()[1:]
            for i, content in enumerate(contents):
                content = content.strip()
                tokens = content.split(' ')
                word = tokens[0]
                vec = tokens[1]
                vec = float(vec)
                word_cap[word] = vec 
    return word_cap

def topic_sim(query, idx2word, t_emb, w_emb):
    '''
        Parameter examples:
        query: 'machine_learning', 'deep_learning'...
        idx2word: [12332: 'delay_spread', 12333: 'data_security', ...
        t_emb: {'machine_learning': array([-0.120063,  0.108274, -0.463706, -0.267065, ...]), ...}
        w_emb: {'delay_spread': array([...]), ...}
    '''
    if query in t_emb:
        q_vec = t_emb[query]
    else:
        q_vec = w_emb[query]
    
    # if len(idx2word) is 5000 (indicating you have 5000 words in your vocabulary) 
    # and you're working with 100-dimensional word embeddings, this line of code 
    # creates a 2D NumPy array with 5000 rows and 100 columns
    word_emb = np.zeros((len(idx2word), 100))

    # for each word in the vocabulary, assign its corresponding word embedding
    # This populates the word_emb array with word embeddings based on the indices in idx2word.
    for i in range(len(idx2word)):
        word_emb[i] = w_emb[idx2word[i]]

    # calculates the dot product between two NumPy arrays: word_emb and q_vec. Here's what's happening:
    # np.dot(word_emb, q_vec): This is using NumPy's dot function to compute the dot product between the
    # 2D array word_emb and the 1D array q_vec. In linear algebra, the dot product of two vectors is a
    # scalar quantity obtained by multiplying corresponding elements and summing the results.

    # If word_emb is a 2D array of shape (n, m) where n is the number of words and m is the dimensionality
    # of the word embeddings, and q_vec is a 1D array of shape (m,), then the dot product operation calculates
    # a new 1D array of shape (n,)

    # this .dotProduct is used to measure the similarity or relevance between the word represented by 'q_vec'
    # and the words represented by the rows of 'word_emb'. 

    #       Semantic Similarity:
    # - If q_vec represents the embedding of a query word, and word_emb represents embeddings of a set of words
    # (for instance, in a vocabulary), the dot product measures the semantic similarity between the query word
    # and each word in the vocabulary.
    # - Higher dot product values indicate that the word represented by the corresponding row in word_emb is
    # more semantically similar to the query word represented by q_vec.

    #       Word Similarity:
    # In word similarity tasks, word_emb could represent embeddings of pairs of words. The goal might be to 
    # measure the similarity between the pairs.
    # Computing dot products between the pairs of word embeddings allows you to quantify their similarity.
    # Higher dot product values imply that the words in the pairs are more similar according to the embedding space.

    res = np.dot(word_emb, q_vec)

    # np.linalg.norm(word_emb, axis=1): This calculates the L2 norm of each row in the word_emb array. 
    # The axis=1 argument specifies that the norm should be calculated along rows. As a result, this
    # operation produces a 1D array where each element represents the L2 norm of the corresponding row in word_emb.

    # res / np.linalg.norm(word_emb, axis=1): This performs element-wise division between the res array
    # (which contains the dot products between q_vec and rows of word_emb) and the 1D array of L2 norms
    # calculated from word_emb.

    # The purpose of this operation is to normalize the dot products, effectively transforming them into
    # cosine similarities. When two vectors are normalized (i.e., divided by their L2 norm), their dot
    # product represents the cosine of the angle between them. In NLP tasks, cosine similarity is often
    # used because it measures the similarity between vectors regardless of their magnitude, focusing only
    # on their direction in the vector space.

    # After this operation, res contains cosine similarity scores between the query vector represented by
    # q_vec and each word or document represented by the rows of word_emb. Higher values in res indicate
    # higher cosine similarity and, consequently, higher similarity between the query vector and the corresponding
    # words or documents.
    res = res/np.linalg.norm(word_emb, axis=1)

    # returns array of indices of the sorted results in descending order
    # if res was [0.8, 0.2, 1.0, 0.5], np.argsort(-res) would return [2, 0, 3, 1]
    sort_id = np.argsort(-res)

    return sort_id

def rank_cap(cap, idx2word, class_name):
    word_cap = np.zeros(len(idx2word))
    for i in range(len(idx2word)):
        if idx2word[i] in cap:
            word_cap[i] = (cap[idx2word[i]]-cap[class_name]) ** 2
        else:
            word_cap[i] = np.array([1.0])
    low2high = np.argsort(word_cap)
    return low2high

def rank_cap_customed(cap, idx2word, class_idxs):
    target_cap = np.mean([cap[idx2word[ind]] for ind in class_idxs])
    word_cap = np.zeros(len(idx2word))
    for i in range(len(idx2word)):
        if idx2word[i] in cap:
            word_cap[i] = (cap[idx2word[i]]-target_cap) ** 2
        else:
            word_cap[i] = np.array([1.0])
    low2high = np.argsort(word_cap)

    return low2high, target_cap

def aggregate_ranking(sim, cap, word_cap, topic, idx2word, pretrain, ent_sent_index, target=None):
    simrank2id = np.ones(len(sim)) * np.inf
    caprank2id = np.ones(len(sim)) * np.inf
    for i, w in enumerate(sim[:]):
        simrank2id[w] = i + 1
    for i, w in enumerate(cap):
        if pretrain == 0:
            if target is not None and word_cap[idx2word[w]] > target:
                caprank2id[w] = i + 1
            if target is None:
                caprank2id[w] = i + 1
    if pretrain == 0:        
        agg_rank = simrank2id * caprank2id
        final_rank = np.argsort(agg_rank)
        final_rank_words = [idx2word[idx] for idx in final_rank[:500] if idx2word[idx] in ent_sent_index]
    else:
        agg_rank = simrank2id
        final_rank = np.argsort(agg_rank)
        final_rank_words = [idx2word[idx] for idx in final_rank[:500] if idx2word[idx] in ent_sent_index]
    # print(final_rank_words)  
    return final_rank_words


def get_common_ent_for_list(l, ent_ent_index):

    parent_cand = set()

    for test_topic in l:
        if len(parent_cand) == 0:
            parent_cand = ent_ent_index[test_topic]
        else:
            parent_cand = parent_cand.intersection(ent_ent_index[test_topic])

    return parent_cand

def get_common_ent_for_list_with_dict(l,d):
    parent_result = set()
    for test_topic in l:
        if len(parent_result) == 0:
            parent_result = set(d[test_topic])
        else:
            parent_result = parent_result.intersection(set(d[test_topic]))

    return parent_result

def get_threshold_from_dict(d, thre):
    parent_result_entities = defaultdict(int)
    for topic in d:
        for ent in d[topic]:
            parent_result_entities[ent] += 1
    # print(parent_result_entities)
    parent_result_entities = [x for x in parent_result_entities if parent_result_entities[x] >= len(d)*thre]
    # print(parent_result_entities)
    return parent_result_entities

def loadEnameEmbedding(filename, dim=100, header=False):
    """ Load the entity embedding with word as context

    :param filename:
    :param dim: embedding dimension
    :return:
    """

    M = dim  # dimensionality of embedding
    ename2embed = {}
    with open(filename, "r") as fin:
        if header:
            next(fin)
        for line in fin:
            seg = line.strip().split()
            word = seg[0:-M]
            del seg[1:-M]
            seg[0] = '_'.join( word )
            embed = np.array([float(ele) for ele in seg[1:]])
            ename2embed[seg[0]] = embed.reshape(1, M)


    return ename2embed

def save_tree_to_file(topic_hier, fileName):
    with open(fileName, 'w') as fout:
        for k in topic_hier:
            fout.write(k)
            for v in topic_hier[k]:
                fout.write('\t'+v)
            fout.write('\n')

            
