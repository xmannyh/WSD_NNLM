import numpy as np
from nltk.stem import WordNetLemmatizer
import pickle
import sys
import nltk
from nltk import FreqDist
import os

################################################
# globals
end_token   = '#END#'
start_token = '#START#'
unk_token   = '#UNK#'
miss_token  = '#MISS#'
tokens = [end_token, start_token, unk_token]
max_sent = 12
folder = './data/processed/'
################################################

wnl = WordNetLemmatizer()

dir_name = './data/masc/spoken/'

def proc_wsd_data():
    vocab_file = "full_vocab_idx_1.p"
    verbs_file = "verbs.p"

    verbs = pickle.load(open(folder + verbs_file, "rb"))
    vocab_idx = pickle.load(open(folder + vocab_file,'rb'))

    para = 'PARAGRAPH_BREAK'
    sent = 'SENTENCE_BREAK'
    nbrk = 'NO_BREAK'

    sentences = []
    senses = []
    for filename in os.listdir(dir_name):
        filename = dir_name+filename

        import xml.etree.ElementTree as ET
        tree = ET.parse(filename)
        root = tree.getroot()

        sent = []
        sense = []
        for child in root:
            break_level = child.attrib['break_level']
            
            if break_level == para or break_level == sent:
                sent = [start_token] + sent + [end_token]
                if len(sent) > max_sent:
                    sent = []
                else:
                    for verb in verbs:
                        if verb in sent:
                            sentences.append(sent)
                            senses.append(sense)
                            break
                sent = []
                            
            txt = wnl.lemmatize(child.attrib['text'])
            if 'sense' in child.attrib:
                sense.append(child.attrib['sense'])
            else:
                sense.append('NONE')
            sent.append(txt)
    n_sent = len(sentences)
    sents = np.ones((n_sent, max_sent), dtype=int) * vocab_idx[end_token]

    senses_set = set([item for sublist in senses for item in sublist])
    senss = np.ones((n_sent, max_sent), dtype=int) * len(senses_set)
    senses_dict = {x: i for i, x in enumerate(senses_set)}

    verb_sents = {}
    for i in range(n_sent):
        for j, word in enumerate(sentences[i]):
            if word in verbs:
                if word in verb_sents:
                    verb_sents[word].append(i)
                else:
                    verb_sents[word] = [i]

            if senses[i][j] != 'NONE':
                senss[i][j] = senses_dict[senses[i][j]]

            if word in vocab_idx:
                sents[i][j] = vocab_idx[word]
            else:
                sents[i][j] = vocab_idx[unk_token]

    n_verb_sents = 0
    for x in verb_sents:
        n_verb_sents+= len(verb_sents[x])
        print('{}: {}'.format(x, len(verb_sents[x])))
    print('Total sentence count: {}'.format(n_verb_sents))

    verb_sentences = np.ones((n_verb_sents, max_sent), dtype=int) * vocab_idx[end_token]
    verb_senses = np.ones((n_verb_sents, max_sent), dtype=int) * len(senses_set)
    current_verb_sents_idx = 0
    for verb in verb_sents:
        for sent_idx in verb_sents[verb]:
            for i in range(len(sents[sent_idx])):
                verb_sentences[current_verb_sents_idx][i] = sents[sent_idx][i]
                verb_senses[current_verb_sents_idx][i] = senss[sent_idx][i]

            current_verb_sents_idx+=1

    print(len(sentences))
    print(verbs)
    print(senss.shape)
    print(sents.shape)
    print(len(senses_dict))
    #np.save(folder + 'wsd_senses.npy', senss)
    #np.save(folder + 'wsd_sents.npy', sents)
    np.save(folder + 'wsd_verb_sentences.npy', verb_sentences)
    np.save(folder + 'wsd_verb_senses.npy', verb_senses)
    pickle.dump(senses_dict, open(folder + 'wsd_senses_idx.p', "wb"))
    return verb_sentences, verb_senses, senses_dict
