from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import csv
import time
import json
import pickle
import argparse

import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

from utils import TextLoader
from model import BasicLSTM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sense_file', type=str, default='wsd_verb_senses.npy',
                        help='location of the senses file')
    parser.add_argument('--test_file', type=str, default="wsd_verb_sentences.npy",
                        help='testing data path')
    parser.add_argument('--save_dir', type=str, default='./save/',
                        help='directory to store checkpointed models')
    parser.add_argument('--output', type=str, default='./prediction.csv',
                        help='output path')
    parser.add_argument('--data_dir', type=str, default='wsd_sents.npy',
                        help='data directory containing input data')
    parser.add_argument('--batch_size', type=int, default=30,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=12,
                       help='RNN sequence length')
    parser.add_argument('--data_set_size', type=int, default=0,
                       help='The number of sentences in the training set.')
    args = parser.parse_args()
    
    # pretty print args
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) 
    
    infer(args)

def infer(args):
    start = time.time()
    
    # Load testing data
    # ====================================
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
        print('restored args:\n', json.dumps(vars(saved_args), indent=4, separators=(',',':'))) 
    
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'rb') as f:
        _, vocab = pickle.load(f)    
    data_loader  = TextLoader(args.test_file, args.sense_file, args.batch_size, args.seq_length, args.data_set_size, shuffle=True)
    
    sense_idx = pickle.load(open('wsd_senses_idx.p','rb'))
    #words to sense dict
    words_sense = pickle.load(open('verbs_sense.p','rb'))

    
    # Predict
    # ===================================
    #checkpoint = tf.train.latest_checkpoint(args.save_dir)

    
    with tf.Graph().as_default():
        with tf.Session() as sess:

            start = time.time()
            saver = tf.train.import_meta_graph('./save2/model.ckpt-65.meta')
            saver.restore(sess,tf.train.latest_checkpoint('./save/'))
            graph = tf.get_default_graph()
            graph_x = graph.get_tensor_by_name("x:0")
            graph_y = graph.get_tensor_by_name("y:0")
            graph_context_layer = graph.get_tensor_by_name("cont_layer:0")
            #graph_softmax_loss = graph.get_operation_by_name("softmax_loss")
            '''
            model = BasicLSTM(saved_args, True)
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)
            '''
            

            data_loader.reset_batch_pointer()
            x_batch_test, y_batch_test, unk_count_test, n_sent_test, senss_test = data_loader.next_batch_test(collect_sense=True)
            feed_dict_test = {graph_x: x_batch_test, graph_y: y_batch_test}
            data_loader.reset_batch_pointer()

            xs = []
            ys = []
            senses = []
            data_loader.reset_batch_pointer()
            for i in range(data_loader.num_batches):
                x_batch, y_batch, unk_count, n_sent, senss = data_loader.next_batch(collect_sense=True, shuffle=False)
                feed_dict = {graph_x: x_batch, graph_y: y_batch}
                wordVecs = sess.run(graph_context_layer, feed_dict)
                
                n_sents = len(y_batch)
                for j in range(n_sents):
                    if y_batch[j] != -1:
                        xs.append(wordVecs[j])
                        ys.append(y_batch[j])
                        senses.append(senss[j])
                    

            #print(xs[0].shape)
            n_words = np.max(ys) + 1
            n_examples = len(ys)
            

            sense_vects = {}
            for i in range(n_examples):
                if senses[i] in sense_vects:
                    sense_vects[senses[i]].append(xs[i])
                else:
                    sense_vects[senses[i]] = [xs[i]]

            sense_keys = sense_vects.keys()

            sense_train_counts = {}
            for key in sense_keys:
                sense_train_counts[key] = len(sense_vects[key][:])
                sense_vects[key] = np.mean(sense_vects[key], axis=0)

            xs_test = []
            ys_test = []
            senses_test= []
            for i in range(data_loader.num_batches):
                
                wordVecs = sess.run(graph_context_layer, feed_dict_test)
                
                n_sents = len(y_batch_test)
                for j in range(n_sents):
                    if y_batch_test[j] != -1:
                        xs_test.append(wordVecs[j])
                        ys_test.append(y_batch_test[j])
                        senses_test.append(senss_test[j])
            
            n_tests = len(ys_test)
            corr = 0
            for i in range(n_tests):
                ambig_word = data_loader.verbs_idx[ys_test[i]]
                ambig_word = data_loader.words[ambig_word]
                correct_sense = senses_test[i]
                if ambig_word in words_sense:
                    max_cos = 0
                    for sense in words_sense[ambig_word]:
                        if sense in sense_idx:
                            sen = sense_idx[sense]
                            if sen in sense_vects:
                                print(cosine_similarity(sense_vects[sen], xs_test[i]))
                                exit()

                loss, contextVecs = sess.run(graph_context_layer, feed_dict)
                print(contextVecs.shape)
                exit()


    print("Saved prediction to {}".format(out_path))
    print("Total run time: {}s".format(time.time() - start))

if __name__ == '__main__':
    main()

