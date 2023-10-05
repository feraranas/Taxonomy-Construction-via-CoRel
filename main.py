import time
import argparse
import numpy as np
import os
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from pytorch_transformers import *
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralCoclustering
from tqdm import tqdm
from random import sample
from utils import *
from model import *
from transfer import *
from co_cluster import *
from train import *
from batch_generation import *
from pytorch_transformers import *
from pytorch_transformers.modeling_bert import *
from transformers import BertTokenizer, BertModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BATCH_SIZE=16
TEST_BATCH_SIZE=512
EPOCHS = 5
max_seq_length = 128

if __name__ == "__main__":

    # This line of code involves the use of the argparse module in Python, 
    # which provides a convenient way to parse command-line arguments.
    parser = argparse.ArgumentParser(description='main',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # this line adds a command-line argument --dataset to the parser. 
    # The --dataset argument is a flag that can be used when running the program from the command line. 
    # If the user doesn't specify a value for --dataset, it defaults to 'dblp'.
    parser.add_argument('--dataset', default='dblp')
    parser.add_argument('--topic_file', default='topics_field.txt')
    parser.add_argument('--out_file', default='keyword_taxonomy.txt')


    # This line of code parses the arguments from the command line and stores them in the args variable.
    #This line parses the command-line arguments using the parse_args() method of the ArgumentParser object
    # (parser). It reads the arguments from the command line, processes them according to the definitions 
    # you specified earlier, and returns an object (args) containing the argument value
    args = parser.parse_args()
    print(args)

    # These lines extract the values of the --dataset and --topic_file arguments from the args object 
    # and store them in variables dataset and topic_file, respectively.
    dataset = args.dataset
    topic_file = args.topic_file


    # ent_sent_index.txt: record the sentence id where each entity occurs; used for generating BERT training sample
    print('------------------loading corpus!------------------')
    ent_sent_index = dict() # initializes an empty dictionary where the data from the file will be stored. 
    with open(dataset+'/ent_sent_index.txt') as f:
        for line in f:
            ent = line.split('\t')[0] #  split the current line into two parts using the tab ('\t') character as a delimiter. 
            tmp = line.strip().split('\t')[1].split(' ') # second part is split into a list of strings using space (' ') as a delimiter.
            tmp = [int(x) for x in tmp] # each element of this list is converted to an integer using a list comprehension. 
            
            # the key is the ent variable (extracted from the first part of the line),
            # and the value is a set of integers obtained from the tmp variable (extracted from the second part of the line). 
            ### Using a set ensures that the data is stored in an unordered collection of unique elements.
            ent_sent_index[ent] = set(tmp) 
    
    # sentences_.txt: sentence id to text
    sentences = dict()
    with open(dataset+'/sentences_.txt') as f:
        # This line iterates through each line of the opened file using a for loop. 
        # The enumerate() function is used here to get both the line content (line) and its index (i).
        # The index i starts from 0 for the first line, 1 for the second line, and so on.
        for i,line in enumerate(f):
            sentences[i] = line

    ent_ent_index = dict()
    with open(dataset+'/ent_ent_index.txt') as f:
        for line in f:
            ent = line.split('\t')[0]
            tmp = line.strip().split('\t')[1].split(' ')
            ent_ent_index[ent] = set(tmp)
    
    print('------------------loading embedding!------------------')

    pretrain = 0
    use_cap0 = False
    file = topic_file.split('_')[1].split('.')[0] # 'field'

    # LOAD WORD EMBEDDING
    # get_emb(vec_file=)
    word_emb, vocabulary, vocabulary_inv, emb_mat = get_emb(vec_file=os.path.join(dataset, 'emb_part_'+file + '_w.txt'))
    
    ### print(word_emb) outputs:
    # 'delay_spread': array([ 0.406334, -0.352674,  0.180382, -0.097917,  0.68063 ,  0.166258]),
    # 'data_security': array([ 0.201929, -0.152651, -0.189683,  0.326593,  0.1081  ,  0.805797]),
    # ...
    # 'adaptive_filter': array([ 7.611800e-02, -7.497800e-02,  1.489310, -3.05475 , 0.224117])}

    ### print(vocabulary) outputs:
    # ['delay_spread': 12332 , 'data_security': 12333, ..., 'adaptive_filter': 16648]

    ### print(vocabulary_inv) outputs inverse of vocabulary:
    # [12332: 'delay_spread', 12333: 'data_security', ..., 16648: 'adaptive_filter']

    ### print(emb_mat) outputs:
    # [[4 5 7], [1 1.01 2.153], ..., [2.14 8.12 9]]


    # LOAD TOPIC EMBEDDING
    # get_temb(vec_file=, topic_file=)
    topic_emb, topic2id, id2topic, topic_hier = get_temb(vec_file=os.path.join(dataset, 'emb_part_'+file+'_t.txt'), topic_file=os.path.join(dataset, topic_file))
    
    ##### print(topic_emb) outputs:
    ## just the 12 seed topic taxonomies embeddings (machine_learning, data_mining, ...)
    # {'data_mining': array([-0.120063,  0.108274, -0.463706, -0.267065, ...]), ... }

    ##### print(topic2id) outputs:
    # {'machine_learning': 0, 'data_mining': 1, 'natural_language_processing': 2, 
    # 'named_entity_recognition': 3, 'information_extraction': 4, 'machine_translation': 5,
    # 'support_vector_machines': 6, 'decision_trees': 7, 'neural_networks': 8,
    # 'association_rule_mining': 9, 'text_mining': 10, 'web_mining': 11}

    ##### print(id2topic) outputs:
    # {0: 'machine_learning', 1: 'data_mining', 2: 'natural_language_processing',
    # 3: 'named_entity_recognition', 4: 'information_extraction',
    # 5: 'machine_translation', 6: 'support_vector_machines',
    # 7: 'decision_trees', 8: 'neural_networks', 9: 'association_rule_mining',
    # 10: 'text_mining', 11: 'web_mining'}

    ##### print(topic_hier) outputs:
    # {'ROOT': ['machine_learning', 'data_mining', 'natural_language_processing'],
    # 'natural_language_processing': ['named_entity_recognition', 'information_extraction', 'machine_translation'],
    # 'machine_learning': ['support_vector_machines', 'decision_trees', 'neural_networks'],
    # 'data_mining': ['association_rule_mining', 'text_mining', 'web_mining']}

    # LOAD WORD SPECIFICITY
    # get_cap(vec_file=)
    word_cap = get_cap(vec_file=os.path.join(dataset, 'emb_part_'+file+'_cap.txt'))
    
    ##### print(word_cap) outputs:
    # makes a dictionary where the key is the word and the value is the word specificity.
    # word_cap = {'can': 0.353891,
    # 'from': 0.353306,
    # ...
    # 'which': 0.256500,
    # 'paper': 0.451159}

    # ename2embed_bert = loadEnameEmbedding(os.path.join(dataset, 'BERTembed.txt'), 768)
    ##### print(ename2embed_bert) outputs:
    # {'delay_spread': array([ 0.406334, -0.352674,  0.180382, -0.097917,  0.68063 ,  0.166258]),
    # apache_solr': array([[ 1.04381144e-01,  1.28338580e-01, -1.25358200e-01,...]), ...}


    print('------------------generating subtopic candidates!------------------')
    # calculate topic representative words: rep_words
    rep_words = {}
    for topic in topic_emb:
        print(topic) # outputs: 'machine_learning', 'data_mining', 'natural_language_processing', ...
        
        # topic: 'machine_learning', ...
        # vocabulary_inv: [12332: 'delay_spread', 12333: 'data_security', ...
        # topic_emb: {'machine_learning': array([-0.120063,  0.108274, -0.463706, -0.267065, ...]), ...}
        # word_emb: {'delay_spread': array([...]), ...}

        sim_ranking = topic_sim(topic, vocabulary_inv, topic_emb, word_emb) 
        if pretrain:
            cap_ranking = np.ones((len(vocabulary))) # creates a 1D Narray equal to # of elements in vocabulary
            word_cap1 = np.ones((len(vocabulary)))
        else:
            # Check this function, might give error
            cap_ranking = rank_cap(word_cap, vocabulary_inv, topic)
        if use_cap0:
            rep_words[topic] = aggregate_ranking(sim_ranking, cap_ranking, word_cap, topic, vocabulary_inv, pretrain, ent_sent_index, word_cap[topic])
        else:
            rep_words[topic] = aggregate_ranking(sim_ranking, cap_ranking, word_cap, topic, vocabulary_inv, pretrain, ent_sent_index)
    rep_words1 = {}
    for topic in topic_emb:
        rep_words1[topic] = [x for x in rep_words[topic]]
    for word in rep_words:
        rep_words[word] = [word]
    # print(len(rep_words1))  # output: machine_learning data_mining natural_language_processing named_entity_recognition
                              # information_extraction machine_translation support_vector_machines decision_trees
                              # neural_networks association_rule_mining text_mining web_mining


    print('------------------initializing relation classifier!------------------')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # print(tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    model = RelationClassifer.from_pretrained('bert-base-uncased')
    # print(model)
    model.float()
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # adjusts the learning rate during training, allowing for more fine-grained control over the optimization process.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)

    print('------------------generating training data!------------------')

    # generating training and testing data
    total_data, sentences_index = process_training_data(sentences, rep_words1, topic_hier, max_seq_length, ent_sent_index, tokenizer)

    # train_data = total_data[:int(len(total_data)/2*0.95)]
    # train_data.extend(total_data[int(len(total_data)/2):int(len(total_data)/2+len(total_data)/2*0.95)])
    # valid_data = total_data[int(len(total_data)/2*0.95):int(len(total_data)/2)]
    # valid_data.extend(total_data[int(len(total_data)/2*0.95+len(total_data)/2):])
    # # test_data = process_test_data(rep_words[test_topic], test_cand, max_seq_length)
    # print(f"training data point number: {len(train_data)}")

    # # training the bert classifier
    # print('------------------training relation classifier!------------------')

    # for epoch in range(EPOCHS):
    #     start_time = time.time()
    #     train_loss, train_acc = train_func(train_data, model, BATCH_SIZE, optimizer, scheduler, generate_batch)
    #     print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        
    #     valid_loss, valid_acc = valid_func(valid_data, model, BATCH_SIZE, generate_batch)
    #     print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
        
        
    #     secs = int(time.time() - start_time)
    #     mins = secs / 60
    #     secs = secs % 60

    #     print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))    


    # print('------------------extracting subtopic candidates!------------------')
    
    # entity_ratio_alltopics = {}
    # entity_count_alltopics = {}

    # training_topic = {}
    # for topic in topic_hier['ROOT']:
    #     if topic in topic_hier:
    #         training_topic[topic] = word_cap[topic]
    # training_topic = sorted(training_topic.items(), key = lambda x: x[1])
    # train_topic = training_topic[0][0]

    # for test_topic in topic_hier['ROOT']:
        
    #     sim_ranking = topic_sim(test_topic, vocabulary_inv, topic_emb, word_emb)
    #     cap_ranking, target_cap = rank_cap_customed(word_cap, vocabulary_inv, [vocabulary[word] for word in topic_hier[train_topic]])
    #     coefficient = max(word_cap[test_topic] / word_cap[train_topic],1)
    #     test_cand = aggregate_ranking(sim_ranking, cap_ranking, word_cap, test_topic, vocabulary_inv, pretrain, ent_sent_index, target_cap*coefficient)

    #     test_data = process_test_data(sentences, rep_words[test_topic], test_cand, max_seq_length,ent_sent_index,ename2embed_bert,  tokenizer)
    #     print(f"test data point number: {len(test_data)}")
    #     if len(test_data) > 10000:
    #         test_data = sample(test_data, 10000)

    #     entity_ratio, entity_count = relation_inference(test_data, model, TEST_BATCH_SIZE)
    #     entity_ratio_alltopics[test_topic] = entity_ratio
    #     entity_count_alltopics[test_topic] = entity_count

    # child_entities_count = sum_all_rel(topic_hier['ROOT'], entity_count_alltopics, mode='child')

    # # print(child_entities_count)

    # child_entities = type_consistent(child_entities_count, ename2embed_bert)

    # print('------------------Topic-Type Matrix Creation!------------------')
    

    # clusters_all = {}
    # k=0
    # start_list = [0]
    # for j,topic in enumerate(topic_hier['ROOT']): 
        
    #     X = []
    #     for ent in child_entities[topic]:
    #         if ent not in word_emb:
    #             continue
    #         X.append(word_emb[ent])
    #     X = np.array(X)

    #     clustering = AffinityPropagation().fit(X)
    #     n_clusters = max(clustering.labels_) + 1
    #     clusters = {}
    #     for i in range(n_clusters):
    #         clusters[str(i)] = [child_entities[topic][x] for x in range(len(clustering.labels_)) if clustering.labels_[x] == i]
            
    #         clusters_all[str(k)] = clusters[str(i)]
    #         k+=1
    #     start_list.append(k)

    # new_clusters = type_consistent_cocluster(clusters_all, ename2embed_bert, n_cluster_min = 8, print_cls = True)

    # tmp = defaultdict(list)

    # print('------------------Subtopics found!------------------')

    # topic_idx = 0
    # for k in range(len(clusters_all)):
    #     if k >= start_list[topic_idx]:
    #         print('\n',topic_hier['ROOT'][topic_idx])
    #         topic_idx += 1
    #     if str(k) in new_clusters and len(new_clusters[str(k)]) > 1:
    #         print(new_clusters[str(k)])
    #         tmp[topic_hier['ROOT'][topic_idx-1]].append(new_clusters[str(k)])

    # child_entities = tmp

    # print('------------------Root Node Candidate Generation!------------------')
    # parent_cand = get_common_ent_for_list(topic_hier['ROOT'],ent_ent_index)
    # if len(parent_cand) > 1000:
    #     parent_cand = type_consistent_for_list(parent_cand, rep_words, ename2embed_bert, False)

    # parent_entity_ratio_alltopics = {}
    # parent_entity_count_alltopics = {}
    # for test_topic in topic_hier['ROOT']:
    #     print(f'test topic: {test_topic}')
        
    #     test_data = process_test_data(sentences, [test_topic], list(parent_cand), max_seq_length,ent_sent_index, ename2embed_bert, tokenizer)
    #     print(f"test data point number: {len(test_data)}")
        
    #     # if len(test_data) > 10000:
    #     #     test_data = sample(test_data, 10000)       
    #     entity_ratio, entity_count = relation_inference(test_data, model, TEST_BATCH_SIZE,mode='child')
    #     parent_entity_ratio_alltopics[test_topic] = entity_ratio
    #     parent_entity_count_alltopics[test_topic] = entity_count

    # parent_entities_count = sum_all_rel(topic_hier['ROOT'], parent_entity_count_alltopics, mode='parent')
    # parent_result = get_threshold_from_dict(parent_entities_count, 1/2)
    # parent_result = type_consistent_for_list(parent_result, rep_words, ename2embed_bert, False)
    # print(f'Discover {len(parent_result)} root nodes!')
    # print(parent_result)
    

    # print('------------------New topic finding!------------------')
    # topic_cand = defaultdict(int)
    # for topic in parent_result:
    #     for ent in ent_ent_index[topic]:
    #         topic_cand[ent] += 1
    # topic_cand = [x for x in topic_cand if topic_cand[x] >= len(parent_result)/2]
    
    # remove_list = []
    # for topic in child_entities_count:
    #     remove_list.extend(child_entities_count[topic])
    # remove_list.extend(parent_result)

    # tmp = []
    # for topic in topic_cand:
    #     if topic not in remove_list:
    #         tmp.append(topic)
    # topic_cand = tmp
    
    # topic_entity_ratio_alltopics = {}
    # topic_entity_count_alltopics = {}
    # for test_topic in parent_result:
    #     print(f'test topic: {test_topic}')        
    #     test_data = process_test_data(sentences, [test_topic], list(topic_cand), max_seq_length,ent_sent_index, ename2embed_bert, tokenizer)
    #     print(f"test data point number: {len(test_data)}")
    #     if len(test_data) > 10000:
    #         test_data = sample(test_data, 10000)    
        
    #     entity_ratio, entity_count = relation_inference(test_data, model, TEST_BATCH_SIZE,mode='child')
    #     topic_entity_ratio_alltopics[test_topic] = entity_ratio
    #     topic_entity_count_alltopics[test_topic] = entity_count

    # topic_entities_count = sum_all_rel(parent_result, topic_entity_count_alltopics, mode='child')


    # topic_entities = get_threshold_from_dict(topic_entities_count, 1/3)
    # cap_list = [word_cap[x] for x in topic_hier['ROOT']]
    # print([(x, word_cap[x]) for x in topic_entities if x in word_cap])
    # topic_entities = get_cap_from_topics(topic_entities, word_cap, cap_list)
    # for t in topic_hier['ROOT']:
    #     if t in topic_hier:
    #         for t1 in topic_hier[t]:
    #             if t1 in topic_entities:
    #                 topic_entities.remove(t1)

    # # topic_entities = [x for x in topic_entities if word_cap[x] < max(cap_list) and word_cap[x] > min(cap_list)]
    # # topic_entities = type_consistent_for_list(topic_entities, rep_words, ename2embed_bert, False)
    # # print(topic_entities)
    # for t in topic_hier['ROOT']:
    #     if t in topic_hier:
    #         for t1 in topic_hier[t]:
    #             if t1 in topic_entities:
    #                 topic_entities.remove(t1)
    #     for t1 in child_entities[t]:
    #         if t1 in topic_entities:
    #             topic_entities.remove(t1)
    # print(topic_entities)


    # print('------------------Subtopic finding for new topics!------------------')
    
    # topic_hier1 = {}

    # topic_hier1['ROOT']= topic_entities
    # for topic in topic_hier:
    #     if topic == 'ROOT':
    #         for t in topic_hier[topic]:
    #             if t not in topic_hier1[topic]:
    #                 topic_hier1[topic].append(t)
    #     else:
    #         topic_hier1[topic] = [x for x in topic_hier[topic]]
    # # print(topic_hier)
    # save_tree_to_file(topic_hier1, 'intermediate.txt')

    # entity_ratio_alltopics1 = {}
    # entity_count_alltopics1 = {}

    # for test_topic in topic_hier1['ROOT']:
    #     if test_topic in topic_hier['ROOT']:
    #         entity_ratio_alltopics1[test_topic] = entity_ratio_alltopics[test_topic]
    #         entity_count_alltopics1[test_topic] = entity_count_alltopics[test_topic]
    #         continue
        
    #     sim_ranking = topic_sim(test_topic, vocabulary_inv, topic_emb, word_emb)
    #     cap_ranking, target_cap = rank_cap_customed(word_cap, vocabulary_inv, [vocabulary[word] for word in topic_hier[train_topic]])
    #     coefficient = max(word_cap[test_topic] / word_cap[train_topic],1)
    #     test_cand = aggregate_ranking(sim_ranking, cap_ranking, word_cap, test_topic, vocabulary_inv, pretrain, ent_sent_index, target_cap*coefficient)
    #     print(f'test topic: {test_topic}')    
    #     test_data = process_test_data(sentences, [test_topic], test_cand, max_seq_length,ent_sent_index, ename2embed_bert, tokenizer)
    #     print(f"test data point number: {len(test_data)}")
        
    #     entity_ratio, entity_count = relation_inference(test_data, model, TEST_BATCH_SIZE)
    #     entity_ratio_alltopics1[test_topic] = entity_ratio
    #     entity_count_alltopics1[test_topic] = entity_count

    # child_entities_count1 = sum_all_rel(topic_hier1['ROOT'], entity_count_alltopics1, mode='child')

    # child_entities1 = type_consistent(child_entities_count1, ename2embed_bert)

    # for ent in topic_hier1['ROOT']:
    #     if ent not in child_entities1:
    #         topic_hier1['ROOT'].remove(ent)

    # clusters_all = {}
    # k=0
    # start_list = [0]
    # for j,topic in enumerate(topic_hier1['ROOT']):   
    #     X = []
    #     for ent in child_entities1[topic]:
    #         if ent not in word_emb:
    #             continue
    #         X.append(word_emb[ent])
    #     if len(X) == 0:
    #         continue
    #     X = np.array(X)

    #     clustering = AffinityPropagation().fit(X)
    #     n_clusters = max(clustering.labels_) + 1
    #     clusters = {}
    #     for i in range(n_clusters):
    #         clusters[str(i)] = [child_entities1[topic][x] for x in range(len(clustering.labels_)) if clustering.labels_[x] == i]
            
    #         clusters_all[str(k)] = clusters[str(i)]
    #         k+=1
    #     start_list.append(k)
    # new_clusters = type_consistent_cocluster(clusters_all, ename2embed_bert, n_cluster_min = 2, print_cls = True, save_file='dblp_field+_cls8')

    # print(start_list)

    # tmp = defaultdict(list)

    # topic_idx = 0
    # for k in range(len(clusters_all)):
    #     if k >= start_list[topic_idx]:
    # #         print('\n',topic_hier1['ROOT'][topic_idx])
    #         topic_idx += 1
    #     if str(k) in new_clusters and len(new_clusters[str(k)]) > 1:
    # #         print(new_clusters[str(k)])
    #         tmp[topic_hier1['ROOT'][topic_idx-1]].append(new_clusters[str(k)])

    # child_entities1 = tmp
    # for t in topic_hier['ROOT']:
    #     child_entities1[t] = child_entities[t]

    # print('------------------Outputing the topical taxonomy!------------------')
        
    # for t in topic_hier1['ROOT']:
    #     if len(child_entities1[t]) == 0:
    #         continue
    #     print(t)
    #     for l in child_entities1[t]:
    #         print(l)
    #     print('')

    # # print the keyword taxonomy, nodes in which will be enriched later by concept learning.
    # with open(os.path.join(dataset, args.out_file), 'w') as fout:
    #     for topic in topic_hier1['ROOT']:  
    #         if len(child_entities1[topic]) > 0:      
    #             fout.write(topic+'\n')
    #             for cls in child_entities1[topic]:
    #                 fout.write(' '.join(cls)+'\n')
    #             fout.write('\n')

    # for topic in topic_hier1['ROOT']:
    #     if len(child_entities1[topic]) > 0:
    #         with open(os.path.join(dataset, 'topics_'+topic+'.txt'),'w') as fout:
    #             for cls in child_entities1[topic]:
    #                 fout.write(' '.join(cls)+'\n')
