# processing training and test data
# use entity pairs to find their co-occurred sentences in the corpus
# and then generate positive and negative training samples

import torch
import numpy as np
import torch.nn.functional as F

def process_training_data(sentences, rep_words, topic_hier, max_seq_length, ent_sent_index, tokenizer):
    '''
    sentences      =>
        sentences[1] = "the trie data_structure has..."
        sentences[2] = "this work investigates o..."
        sentences[n] = "in all cases , the d..."

    rep_words      => # 12 topics in seed taxonomy
        {'data_mining': ['algorithms', 'graphs', 'data_science'...], ...}

    topic_hier     =>
        {'ROOT': ['machine_learning', 'data_mining', 'natural_language_processing'],
        'natural_language_processing': ['named_entity_recognition', 'information_extraction', 'machine_translation'],
        'machine_learning': ['support_vector_machines', 'decision_trees', 'neural_networks'],
        'data_mining': ['association_rule_mining', 'text_mining', 'web_mining']}

    max_seq_length => 128

    ent_sent_index => dictionary with sentence id where each entity occurs (used for generating BERT training sample)
        {'trie' = [0 2 2 3 3 299 ...]
        'algorithms = [2 16 49 50 ...]
        'software' = [0 4 4 6 9 8 ...]}

    tokenizer      => BertTokenizer 'bert_base_uncased'
    '''

    parent_list = [x for x in topic_hier if x != 'ROOT'] # output: ['natural_language_processing', 'machine_learning', 'data_mining'] 
    sentences_index = []
    final_data = []

    real_rep_words = {}
    for key in rep_words:
        real_rep_words[key] = []
        
    print("collecting POSITIVE samples!")
    
    for parent in parent_list:
        # parent_i = 'natural_language_processing', 'machine_learning', 'data_mining'
        # topic_hier[parent_i] = ['named_entity_recognition', 'information_extraction', 'machine_translation']
        # child_i = 'named_entity_recognition', 'information_extraction', 'machine_translation'
        for child in topic_hier[parent]:
#             print(child)
            count = 10
            # b = 'algorithms', 'graphs', 'data_science'...
            for b in rep_words[child]:
                if b not in ent_sent_index:
                    # si la palabra relacionada no la encuentra, simplemente vamos a la siguiente
                    continue
                for a in rep_words[parent]:
                    # si la palabra relacionada no la encuentra, simplemente vamos a la siguiente
                    if a not in ent_sent_index:
                        continue
                    # ent_sent_index[a] = [0 2 2 3 3 299 ...]
                    # ent_sent_index[b] = [2 16 49 50 ...]

                    # intersection() is a set method in Python. When called on a set, it returns a new set containing elements
                    # that are present in both the original set and the set passed as an argument. In this case, it calculates
                    # the intersection of sentence indices between entities a and b.
                    cooccur = ent_sent_index[a].intersection(ent_sent_index[b])
                    # print(cooccur)
                    if len(cooccur) > 0:
                        if a not in real_rep_words[parent]:
                            real_rep_words[parent].append(a)
                        if b not in real_rep_words[child]:
                            real_rep_words[child].append(b)

                        # sen es un index que se generó mediante la intersección de palabras en una oración
                        # para cada oración con dicha intersección la agregamos a 'sentences_index'
                        for sen in cooccur:
                            # This line works for dblp 'sentences_.txt' given that the corpus is only 560_846 lines long
                            if sen > 560_846:
                                continue
                            sentences_index.append(sen)
                            try:
                                s = sentences[sen]
                                # print(s)
                            except KeyError as e:
                                print(f"KeyError: {e}, 'Sen' not found: {sen}")
                            
                            # add a special token [CLS] at the beginning of the string. 
                            # In (NLP) tasks, [CLS] (short for "classification") is a special token used in models like BERT
                            # (Bidirectional Encoder Representations from Transformers) for sentence classification.
                            s = '[CLS] ' + s

                            # After the split operation, the string s is transformed into a list of words:
                            # ['[CLS]', 'this', 'paper', 'relates', 'to', 'a', 'concrete', ...]
                            s = s.split(' ')
                            
                            if a not in s or b not in s:
                                continue

                            # index of the first occurrence of the string [parent word] in the sentence
                            p_index = s.index(a)
                            # index of the first occurrence of the string [child word] in the sentence
                            c_index = s.index(b)

                            s[p_index] = '[MASK]'
                            s[c_index] = '[MASK]'

                            s = ' '.join(s).replace('_', ' ').replace('-lrb-','(').replace('-rrb-',')')
                            
                            # The input_id variable stores the encoded sequence.
                            # this sequence is represented as a list of integers, where each integer corresponds to
                            # a token ID in the vocabulary. These token IDs are what the model uses as input during
                            # training and prediction.
                            input_id = tokenizer.encode(s)

                            # tokenizes the resulting string into a list of tokens without converting them into token IDs
                            # Each element of tokened_text represents a separate token.
                            tokened_text = tokenizer.tokenize(s)

                            # Finding Mask Tokens (Special Tokens with ID 103)
                            # why 103?
                            mask_id = [x for x in range(len(input_id)) if input_id[x] == 103]
                            
                            # print(mask_id)
                            
                            # Handling Cases with Fewer Than 2 Mask Tokens
                            if len(mask_id) < 2:
                                continue
                                # debugging cases
                                # print(' '.join(s))
                                # print(a)
                                # print(b)

                            # Handling Cases Where Input Length Exceeds a Maximum
                            if len(input_id) > max_seq_length:
                                # If the distance between the two mask tokens (mask_id[1] - mask_id[0])
                                # is greater than or equal to the maximum sequence length
                                # the loop continues to the next iteration,
                                # skipping further processing for this particular input.
                                if mask_id[1] - mask_id[0] >= max_seq_length:
                                    continue
                                # If the distance between the mask tokens is less than the maximum sequence length,
                                # the code updates the input_id list to include only the tokens between the two mask tokens. 
                                else:
                                    input_id = input_id[mask_id[0]:mask_id[1]+1]
                                    #  also update p_index and c_index variables based on the positions of the mask tokens.
                                    p_index = 0 if p_index < c_index else mask_id[1] - mask_id[0]
                                    c_index = 0 if c_index < p_index else mask_id[1] - mask_id[0]
                            else:
                                p_index = mask_id[0] if p_index < c_index else mask_id[1] 
                                c_index = mask_id[0] if c_index < p_index else mask_id[1]
                            
                            # input_id = torch.tensor(input_id)
                            input_id = np.array(input_id)
                            input_id = torch.from_numpy(input_id)


                            # r_sentence = F.pad(input_id.detach().clone().requires_grad_(True), (0, max_seq_length - len(input_id)), "constant", 0)
                            

                            # F.pad function from the PyTorch functional (F) module to pad a tensor input_id with zeros to achieve a specified size.
                            # Second parameter: This tuple specifies the amount of padding to be added to the tensor. 
                            # In this case, it pads the tensor with zeros on the right side (appending zeros) to make 
                            # its length equal to max_seq_length. The first element of the tuple, 0, indicates that no 
                            # padding is added to the left side, and the second element, max_seq_length - len(input_id), 
                            # specifies the number of zeros to add on the right side to reach the desired length max_seq_length.

                            # Third parameter: "constant". This parameter specifies the padding mode. In this case, it's set 
                            # to "constant", meaning the padding values are constant and specified by the next parameter.
                            r_sentence = F.pad(input_id,
                                               (0, max_seq_length - len(input_id)),
                                               "constant", 0)
                            
                            # parameters:
                            # torch.ones_like(input_id): This creates a tensor of ones with the same shape as input_id.
                            # Assuming input_id is a tensor representing a sequence, this tensor has ones at the positions where the input sequence exists.

                            # torch.zeros(max_seq_length - len(input_id), dtype=torch.int64): This creates a tensor of zeros 
                            # with a length equal to the difference between max_seq_length and the length of the input_id tensor.
                            # This part of the code creates zeros to pad the attention mask to the desired length.

                            # torch.cat(..., dim=0): This concatenates the tensors created above along dimension 0 (rows).
                            # In this context, it concatenates the tensor of ones representing the existing sequence and the tensor of
                            # zeros representing the padding, creating an attention mask for the entire sequence.

                            # The resulting attention_mask is a binary mask where ones indicate the positions of the original input sequence,
                            # and zeros indicate the positions that are padded. Attention masks are often used in transformer models to mask
                            # out padded tokens during computation, ensuring that the model does not attend to the padded positions,
                            # which do not contain meaningful information.
                            attention_mask = torch.cat((torch.ones_like(input_id),
                                                        torch.zeros(max_seq_length - len(input_id), dtype=torch.int64)),
                                                        dim=0)
                            
                            p_mask = np.zeros((max_seq_length))
                            p_mask[p_index] = 1
                            c_mask = np.zeros((max_seq_length))
                            c_mask[c_index] = 1
                                            
                            final_data.append([r_sentence, p_mask, c_mask, attention_mask, 0])
                            final_data.append([r_sentence, c_mask, p_mask, attention_mask, 1])
                            
                                
    pos_len = len(final_data)
    print(f"positive data number: {pos_len}")

    print("collecting NEGATIVE samples from siblings!")  
    for parent in parent_list:
        # parent_i = 'natural_language_processing', 'machine_learning', 'data_mining'
        # topic_hier[parent_i] = ['named_entity_recognition', 'information_extraction', 'machine_translation']
        # child_i = 'named_entity_recognition', 'information_extraction', 'machine_translation'
        for child in topic_hier[parent]:
#             print(child)
            count = 10
            # b is every related word to child_i
            for b in rep_words[child]:
                if b not in ent_sent_index:
                    continue
                for a in rep_words[child]:
                    if a == b:
                        continue
                    if a not in ent_sent_index:
                        continue
                    cooccur = ent_sent_index[a].intersection(ent_sent_index[b])
                    if len(cooccur) > 0:
                        # For each co-occurence of word a & b in the same sentence
                        for sen in cooccur:
                            # For the case of dblp not having enough sentences.
                            if sen > 560_846:
                                continue

                            if sen in sentences_index:
                                continue
                            if np.random.random(1) > 0.1:
                                continue
                            sentences_index.append(sen)
                            s = sentences[sen]

                            s = '[CLS] ' + s

                            s = s.split(' ')
                            if a not in s or b not in s:
                                continue
                            p_index = s.index(a)
                            c_index = s.index(b)
                            s[p_index] = '[MASK]'
                            s[c_index] = '[MASK]'
                            s = ' '.join(s).replace('_', ' ').replace('-lrb-','(').replace('-rrb-',')')
                            
                            input_id = tokenizer.encode(s)
                            mask_id = [x for x in range(len(input_id)) if input_id[x]==103]                                                       

                            if len(input_id) > max_seq_length:
                                if mask_id[1] - mask_id[0] >= max_seq_length:
                                    continue
                                else:
                                    input_id = input_id[mask_id[0]:mask_id[1]+1]
                                    p_index = 0 if p_index<c_index else mask_id[1] - mask_id[0]
                                    c_index = 0 if c_index<p_index else mask_id[1] - mask_id[0]
                            else:
                                p_index = mask_id[0] if p_index<c_index else mask_id[1] 
                                c_index = mask_id[0] if c_index<p_index else mask_id[1]
                            

                            input_id = np.array(input_id)
                            input_id = torch.from_numpy(input_id)

                            r_sentence = F.pad(input_id,(0,max_seq_length-len(input_id)), "constant", 0)
                            attention_mask = torch.cat((torch.ones_like(input_id), 
                                                        torch.zeros(max_seq_length-len(input_id), dtype=torch.int64)), dim=0)
                            
                            p_mask = np.zeros((max_seq_length))
                            p_mask[p_index] = 1
                            c_mask = np.zeros((max_seq_length))
                            c_mask[c_index] = 1
                                            
                            final_data.append([r_sentence, p_mask, c_mask, attention_mask, 2])
                            

    print(len(final_data))
    
    print("collecting NEGATIVE samples from corpus!")
    while len(final_data) < pos_len * 2:
        if len(final_data) % 100 == 0:
            print(pos_len)
            print(len(final_data))
        sen = np.random.choice(len(sentences))
        # remove positive sentences
        if sen in sentences_index:
            continue
            
        s = sentences[sen]
        s = '[CLS] '+s
        s = sentences[sen].split(' ')
        
        # randomly choose pairs
        entities = [x for x in s if "_" in x]
        if len(entities) < 2:
            continue
        p_index, c_index = np.random.choice(len(entities),2)
        while c_index == p_index:
            continue
            # c_index = np.random.choice(len(entities))
            
        s[p_index] = '[MASK]'
        s[c_index] = '[MASK]'
        s = ' '.join(s).replace('_', ' ').replace('-lrb-','(').replace('-rrb-',')')
        input_id = tokenizer.encode(s)
        mask_id = [x for x in range(len(input_id)) if input_id[x]==103]                                                       

        if len(input_id) > max_seq_length:
            if mask_id[1] - mask_id[0] >= max_seq_length:
                continue
            else:
                input_id = input_id[mask_id[0]:mask_id[1]+1]
                p_index = 0 if p_index<c_index else mask_id[1] - mask_id[0]
                c_index = 0 if c_index<p_index else mask_id[1] - mask_id[0]
        else:
            p_index = mask_id[0] if p_index<c_index else mask_id[1] 
            c_index = mask_id[0] if c_index<p_index else mask_id[1]

        input_id = np.array(input_id)
        input_id = torch.from_numpy(input_id)

        r_sentence = F.pad(input_id,(0,max_seq_length-len(input_id)), "constant", 0)

        attention_mask = torch.cat((torch.ones_like(input_id), torch.zeros(max_seq_length-len(input_id), dtype=torch.int64)),dim=0)

        p_mask = np.zeros((max_seq_length))
        p_mask[p_index] = 1
        c_mask = np.zeros((max_seq_length))
        c_mask[c_index] = 1

        final_data.append([r_sentence, p_mask, c_mask, attention_mask, 2])
        
    return final_data, sentences_index


# def process_test_data(sentences, test_topic_rep_words, test_cand, max_seq_length, ent_sent_index, ename2embed_bert, tokenizer):

    # print("collecting positive samples!")
#     final_data = []
    

#     count = 10
#     for b in test_topic_rep_words:
#         if b not in ent_sent_index or b not in ename2embed_bert:
#             continue
#         for a in test_cand:
#             if a not in ent_sent_index or a not in ename2embed_bert:
#                 continue
#             if a == b:
#                 continue
#             cooccur = ent_sent_index[a].intersection(ent_sent_index[b])
#             if len(cooccur) > 0:
#                 for sen in cooccur:
#                     s = sentences[sen]
#                     s = '[CLS] '+s
#                     s = s.split(' ')
#                     if a not in s or b not in s:
#                         continue
#                     p_index = s.index(a)
#                     c_index = s.index(b)
#                     s[p_index] = '[MASK]'
#                     s[c_index] = '[MASK]'
#                     s = ' '.join(s).replace('_', ' ').replace('-lrb-','(').replace('-rrb-',')').split(' ')
#                     input_id = tokenizer.encode(' '.join(s))
#                     mask_id = [x for x in range(len(input_id)) if input_id[x]==103]


#                     if len(input_id) > max_seq_length:
#                         if mask_id[1] - mask_id[0] >= max_seq_length:
#                             continue
#                         else:
#                             input_id = input_id[mask_id[0]:mask_id[1]+1]
#                             p_index = 0 if p_index<c_index else mask_id[1] - mask_id[0]
#                             c_index = 0 if c_index<p_index else mask_id[1] - mask_id[0]
#                     else:
#                         p_index = mask_id[0] if p_index<c_index else mask_id[1] 
#                         c_index = mask_id[0] if c_index<p_index else mask_id[1]

#                     input_id = torch.tensor(input_id)
#                     r_sentence = F.pad(torch.tensor(input_id),(0,max_seq_length-len(input_id)), "constant", 0)
#                     attention_mask = torch.cat((torch.ones_like(input_id), torch.zeros(max_seq_length-len(input_id), dtype=torch.int64)),dim=0)

#                     p_mask = np.zeros((max_seq_length))
#                     p_mask[p_index] = 1
#                     c_mask = np.zeros((max_seq_length))
#                     c_mask[c_index] = 1

#                     final_data.append([r_sentence, p_mask, c_mask, attention_mask, a])
#                     final_data.append([r_sentence, c_mask, p_mask, attention_mask, a])

    
    
#     return final_data

# def generate_batch(batch):
#     input_ids = torch.tensor([np.array(entry[0]) for entry in batch])
#     entity1_mask = torch.tensor([entry[1] for entry in batch])
#     entity2_mask = torch.tensor([entry[2] for entry in batch])
#     attention_mask = torch.tensor([np.array(entry[3]) for entry in batch])
#     labels = torch.tensor([entry[4] for entry in batch])
#     return input_ids, entity1_mask, entity2_mask, attention_mask, labels 

# def generate_test_batch(batch):
#     input_ids = torch.tensor([np.array(entry[0]) for entry in batch])
#     entity1_mask = torch.tensor([entry[1] for entry in batch])
#     entity2_mask = torch.tensor([entry[2] for entry in batch])
#     attention_mask = torch.tensor([np.array(entry[3]) for entry in batch])
#     entity = [entry[4] for entry in batch]
#     return input_ids, entity1_mask, entity2_mask, attention_mask, entity