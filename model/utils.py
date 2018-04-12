import os
import re
import csv
import glob
import pickle
import string
import collections

import numpy as np
import gensim

################################################
# globals
end_token   = '#END#'
start_token = '#START#'
unk_token   = '#UNK#'
miss_token  = '#MISS#'
tokens = [end_token, start_token, unk_token]
################################################

class TextLoader():
    def __init__(self, data_dir, sense_file, batch_size, seq_length, data_set_size, shuffle=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        vocab_file = data_dir + "full_idx_vocab_1.p"
        #data_file = "full_sents_1.npy"
        data_file = data_dir + "full_sents_1.npy"
        verbs_file = data_dir + "verbs.p"
        sense_file = data_dir + sense_file

        print("Loading preprocessed files")
        self.load_preprocessed(vocab_file, sense_file, data_file, verbs_file, data_set_size, shuffle)
        self.reset_batch_pointer()

    def next_batch_test(self, set_to_choose=0, collect_sense=False):
        if set_to_choose == 0:
            #choose dev
            self._data = self.dev_set
            num_data = len(self.dev_set)
        else:
            #choose test
            self._data = self.test_set
            num_data = len(self.test_set)

        num_batches = num_data // self.batch_size

        if self.pointer + self.batch_size > num_data:
            self.pointer = 0
        
        x = self._data[self.pointer:self.pointer+self.batch_size, 0:-1]
        sense = self.sense_dev_set[self.pointer:self.pointer+self.batch_size, 0:-1]
        end_idx = self.vocab[end_token]

        
        unk_count = np.where(x == self.vocab[unk_token])[0].shape[0]


        y = []
        ends = []
        senss = []
        # print(x.shape)
        for i in range(x.shape[0]):
            for j in range(len(x[i])):
                if x[i][j] == end_idx:
                    ends.append(j)
                    break
            candidates = []
            for verb in self.verbs_idx:
                if verb in x[i]:
                    candidates.append(verb)
            
            #get the verb to remove and for the model to try and predict
            if len(candidates) > 0:
                missing_verb = np.random.randint(len(candidates))
                missing_verb = candidates[missing_verb]
                missing_verb_idx = np.where(x[i] == missing_verb)[0][0]
                y.append(self.verbs_idx.index(missing_verb))

                if collect_sense == True:
                    senss.append(sense[i][missing_verb_idx])
                x[i][missing_verb_idx] = self.vocab[miss_token]
            else:
                missing_verb = 0
                #an error happened
                print('##################### an error happened(test) #####################')
                y.append(-1)
                if collect_sense == True:
                    senss.append(-1)

        y = np.array(y)
        self.pointer += self.batch_size
        if len(ends) == 0:
            avg_end = 0
        else:
            avg_end = int(np.mean(ends))
        return x, y, unk_count, avg_end, senss

    def map_id(self, sentences):
        data = []
        for s in sentences:
            id_s = np.ones(self.seq_length, np.int32) * self.vocab[end_token]
            for i, word in enumerate(s.split()):
                id_s[i] = self.vocab.get(word, 0)
            data.append(id_s)

        data = np.array(data)

        return data

    def load_preprocessed(self, vocab_file, sense_file, data_file, verbs_file, data_set_size, shuffle=False):
        with open(vocab_file, 'rb') as f:
            #idx->words
            self.words = pickle.load(f)
        with open(verbs_file, 'rb') as f:
            self.verbs = pickle.load(f)
        self.vocab_size = len(self.words)
        #words->idx
        self.vocab = dict(zip(self.words, range(len(self.words))))
        self.verbs_idx = []
        for verb in self.verbs:
            self.verbs_idx.append(self.vocab[verb])
        
        self.dataset  = np.load(data_file)
        self.senses = np.load(sense_file)
        if data_set_size == 0:
            data_set_size = len(self.dataset)
        if shuffle == True:
            print('shuffle')
            perm = np.arange(data_set_size)
            np.random.shuffle(perm)
            self.dataset = self.dataset[perm]
            self.senses = self.senses[perm]
        #create dev and test sest partition borders
        train_end = int(data_set_size*0.8)
        dev_end    = train_end+int(data_set_size*0.1)
        test_start = dev_end
        test_end   = test_start+int(data_set_size*0.1)
        # create dev, test, and training set
        self.dev_set  = self.dataset[train_end:dev_end]
        self.test_set = self.dataset[test_start:test_end]
        self.data     = self.dataset[:train_end]
        self.sense_dev_set  = self.senses[train_end:dev_end]
        self.sense_test_set = self.senses[test_start:test_end]
        self.sense_data     = self.senses[:train_end]

        self.num_data = len(self.data)
        self.num_batches = self.num_data // self.batch_size
        self.num_batches_test = len(self.test_set) // self.batch_size

    def next_batch(self, shuffle=True, collect_sense=False):
        if self.pointer + self.batch_size > self.num_data:
            self.pointer = 0
        
        if self.pointer == 0 and shuffle:
            perm = np.arange(self.num_data)
            np.random.shuffle(perm)
            self._data = self.data[perm]

        elif self.pointer == 0 and shuffle == False:
            self._data = self.data
        
        #x, y = self._data[self.pointer:self.pointer+self.batch_size, 0:-1], self._data[self.pointer:self.pointer+self.batch_size, 1: ]
        x = self._data[self.pointer:self.pointer+self.batch_size, 0:-1]
        sense = self.sense_data[self.pointer:self.pointer+self.batch_size, 0:-1]
        end_idx = self.vocab[end_token]

        
        unk_count = np.where(x == self.vocab[unk_token])[0].shape[0]


        y = []
        senss = []
        ends = []
        for i in range(x.shape[0]):
            for j in range(len(x[i])):
                if x[i][j] == end_idx:
                    ends.append(j)
                    break
            candidates = []
            for verb in self.verbs_idx:
                if verb in x[i]:
                    candidates.append(verb)
            
            #get the verb to remove and for the model to try and predict
            if len(candidates) > 0:
                missing_verb = np.random.randint(len(candidates))
                missing_verb = candidates[missing_verb]
                missing_verb_idx = np.where(x[i] == missing_verb)[0][0]

                y.append(self.verbs_idx.index(missing_verb))
                if collect_sense == True:
                    senss.append(sense[i][missing_verb_idx])
                x[i][missing_verb_idx] = self.vocab[miss_token]
            else:
                #an error happened
                print('##################### an error happened(train) #####################')
                y.append(-1)
                if collect_sense == True:
                    senss.append(-1)

            


        y = np.array(y)
        self.pointer += self.batch_size
        return x, y, unk_count, int(np.mean(ends)), senss

    def reset_batch_pointer(self):
        self.pointer = 0


def get_vocab_embedding(save_dir, words, embedding_file):
    matrix_file = os.path.join(save_dir, 'embedding.npy')
    
    if os.path.exists(matrix_file):
        print("Loading embedding matrix")
        embedding_matrix = np.load(matrix_file)
    else:
        print("Building embedding matrix")
        word_vectors = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=True)
        dim = word_vectors['word'].size
        embedding_matrix = np.zeros(shape=(len(words), dim), dtype='float32')
        
        for i, word in enumerate(words):
            # '<UNK>'
            if i == 0:
                continue
            else:
                if word in word_vectors:
                    embedding_matrix[i] = word_vectors[word]
                else:
                    embedding_matrix[i] = np.random.uniform(-0.25,0.25,dim)

        np.save(matrix_file, embedding_matrix)

    return embedding_matrix


class TestLoader():
    def __init__(self, input_file, vocab_dict, seq_length):
        self.input_file = input_file
        self.vocab_dict = vocab_dict
        self.seq_length = seq_length

        test_dir = os.path.dirname(input_file)
        sent_file = os.path.join(test_dir, "test_sent.pkl")
        test_file = os.path.join(test_dir, "test_data.pkl")

        if not (os.path.exists(sent_file) and os.path.exists(test_file)):
            print("Reading testing file")
            self.preprocess(input_file, sent_file, test_file)
        else:
            print("Loading preprocessed testing file")
            self.load_preprocessed(sent_file, test_file)
        
    
    def clean_str(self, string):
        string = re.sub(r"\[([^\]]+)\]", " ", string)
        string = re.sub(r"\(([^\)]+)\)", " ", string)
        string = re.sub(r"[^A-Za-z0-9,!?.;]", " ", string)
        string = re.sub(r"\.", " . ", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r";", " ; ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def map_id(self, test_sent):
        for instance in test_sent:
            candidates = instance['candidates']
            encoded_cand = []
            for sent in candidates:
                # encode a sentence to 40 words
                encoded_sent = np.ones(self.seq_length, np.int32) * self.vocab_dict['<END>'] 
                for i, word in enumerate(sent.split()):
                    if i >= self.seq_length:
                        break
                    encoded_sent[i] = self.vocab_dict.get(word, 0)
                encoded_cand.append(encoded_sent)
            instance['candidates'] = encoded_cand

        return test_sent

    def preprocess(self, input_file, sent_file, test_file):
        test_sent = []
        keys = ['a)', 'b)', 'c)', 'd)', 'e)']
        with open(input_file, "r", encoding='latin1') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                test_instance = {'id':row['id'], 'candidates':[] }
                question = row['question']
                choices = [row[x] for x in keys]
                questions = [question.replace('_____', word) for word in choices]
                test_instance['candidates'] = ['<START> ' + self.clean_str(q) + ' <END>'
                                                for q in questions]
                test_sent.append(test_instance)

        self.sentences = test_sent 
        with open(sent_file, 'wb') as f:
            pickle.dump(test_sent, f)

        self._data = self.map_id(test_sent)
        with open(test_file, 'wb') as f:
            pickle.dump(self._data, f)

    def load_preprocessed(self, sent_file, test_file):
        with open(sent_file, 'rb') as f:
            self.sentences = pickle.load(f)
        with open(test_file, 'rb') as f:
            self._data = pickle.load(f)
    
    def get_data(self):

        return self._data
    
        
