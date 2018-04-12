from nltk.stem import WordNetLemmatizer
import sys
import os
import nltk
from nltk import FreqDist
import pickle

def generate_verbs(dir_name, num_verbs, folder='./data/processed/'):
    wnl = WordNetLemmatizer()
    #dir_name = sys.argv[1]

    verbs = {}
    verbs_sense = {}

    #dir_name = './data/masc/written/essays/'
    for filename in os.listdir(dir_name):
        if filename[-1] is 'l':
            pickle_name = "freq_verbs_" + filename.replace('.xml','') + ".p"
            filename = dir_name+filename

            import xml.etree.ElementTree as ET
            tree = ET.parse(filename)
            root = tree.getroot()

            for child in root:
                if 'sense' in child.attrib and 'pos' in child.attrib:
                    word = wnl.lemmatize(child.attrib['text'])
                    if word not in verbs:
                        verbs[word] = 1
                        verbs_sense[word] = set([child.attrib['sense']])
                    else:
                        verbs[word] += 1
                        verbs_sense[word] = verbs_sense[word] | set([child.attrib['sense']])

    all_verbs = verbs.copy()
    all_verbs_sense = verbs_sense.copy()

    keys = list(verbs.keys())
    for verb in keys:
        if len(verbs_sense[verb]) < 2 or verbs[verb] < 40:
            del verbs[verb]
            del verbs_sense[verb]

    verbs = FreqDist(verbs)
    verbs = verbs.most_common(num_verbs)
    for verb in verbs:
        print('({}) {}: {}'.format(verb[1], verb[0], len(verbs_sense[verb[0]])))


    verbs = list(dict(verbs).keys())
    pickle.dump(all_verbs, open(folder + 'verbs_all.p', "wb"))
    pickle.dump(all_verbs_sense, open(folder + 'verbs_sense_all.p', "wb"))
    pickle.dump(verbs, open(folder + 'verbs.p', "wb"))
    pickle.dump(verbs_sense, open(folder + 'verbs_sense.p', "wb"))

