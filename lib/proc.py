from nltk.stem import WordNetLemmatizer
import time
import sys
import pickle
import numpy as np
import collections

################################################
# globals
end_token   = '#END#'
start_token = '#START#'
unk_token   = '#UNK#'
miss_token  = '#MISS#'
tokens = [end_token, start_token, unk_token]

#max sentence word count
max_sent   = 12
folder = './data/processed/'
################################################
# contains punctuation
def proc(s):
    p1 = ['!','\"','&','\'','(',')','?','@','.','_']
    p2 = ['*','+',',','-','--','/',':',';','>','`']
    punct = p1 + p2

    num = ['1','2','3','4','5','6','7','8','9','0']

    s = s.split()

    if 'a.m.' in s:
        s[s.index('a.m.')] = 'x' + 'a.m.'
        del s[s.index('xa.m.')-1]

    if 'p.m.' in s:
        s[s.index('p.m.')] = 'x' + 'p.m.'
        del s[s.index('xp.m.')-1]

    for n in num:
        for w in s:
            if end_token not in w:
                if n in w:
                    s[s.index(w)] = 'numbr'

    wnl = WordNetLemmatizer()
    n_s = len(s)
    dels = []
    for i in range(n_s):
        lemma = str(wnl.lemmatize(s[i]))
        if lemma is not s[i]:
            #tag = 'lemma=' + lemma
            s[i] = lemma

        for sym in punct:
            if s[i] in tokens:
                continue
            elif sym in s[i]:
                s[i] = s[i].replace(sym,'')
                if len(s[i]) == 0:
                    dels.append(i)
                    continue
                #word = word.replace(sym,'')
    
    for idx in dels[::-1]:
        del s[idx]
    return s
################################################
# 
def build_vocab(sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        words = []
        for s in sentences:
            words += s
        word_counts = collections.Counter(words)
        # Mapping from index to word
        # might want to lower the hit rate... (100)
        idx_vocab = [x[0] for x in word_counts.most_common() if x[1] > 25]
       
        idx_vocab = list(sorted(idx_vocab))
        idx_vocab = [miss_token, unk_token] + idx_vocab
        
        
        # Mapping from word to index
        vocab_idx = {x: i for i, x in enumerate(idx_vocab)}
        return vocab_idx, idx_vocab

def clean(x, verbs, vocab_idx):
    for i in range(x.shape[0]):
        candidates = []
        for verb in verbs:
            if vocab_idx[verb] in x[i]:
                candidates.append(verb)
        
        #get the verb to remove and for the model to try and predict
        if len(candidates) == 0:
            print('uhoh')

################################################
#
def from_text(filename, verbs):
    vocab      = {} #vocab counts
    ex         = 0  #number of skipped sentences
    
    with open(filename) as f:
        doc = f.readlines()
        f.close()
    print('file opened and closed')

    num_sent = len(doc)
    #num_sent = 100000
    sentences = []
    n_verbs = len(verbs)
    verbs_count = np.zeros(n_verbs, dtype=int)
    current_verb = 0
    saved_sents = 0
    for i in range(num_sent):
        if saved_sents >= 20000:
            break
        if saved_sents % 1000 == 0:
            print('saved {}'.format(saved_sents))
        if i % 100000 == 0:
            print('processed {}'.format(i))
        try:
            sent = doc[i]
            if sent[len(sent)-1] is '\n':
                sent = sent[:len(sent)-1]
            else:
                sent = sent[:len(sent)]

            sent = proc(sent)
            sent = [start_token] + sent + [end_token]
            if len(sent) > max_sent:
                ex+=1
                continue

            # redirecting into > output.txt
            #sent = ' '.join(sent)
            #print(sent)
            for current_verb in range(len(verbs)):
                if np.all([x == 5 for x in verbs_count]):
                    verbs_count[:] = 0
                    
                if verbs_count[current_verb] > 4:
                    continue

                if verbs[current_verb] in sent:
                    sentences.append(sent)
                    verbs_count[current_verb] += 1
                    saved_sents+= 1
                    break

            for s in sent:
                if s in vocab:
                    vocab[s] += 1
                else:
                    vocab[s] = 1
        except UnicodeDecodeError:
            ex += 1
            continue
    del doc
    print('sentences processed')

    print('building dicts')
    vocab_idx, idx_vocab = build_vocab(sentences)
    sents = np.ones((len(sentences), max_sent), dtype=int) * vocab_idx[end_token]
    n_sent = len(sentences)
    for i in range(n_sent):
        for j, word in enumerate(sentences[i]):
            if word in vocab_idx:
                sents[i][j] = vocab_idx[word]
            else:
                sents[i][j] = vocab_idx[unk_token]
    print('dicts built')
    #sents = clean(sents)
    return vocab, vocab_idx, idx_vocab, sents

################################################

start_time = time.time()


verbs_file = "verbs.p"
verbs = pickle.load(open(folder + verbs_file, 'rb'))
verbs_dict = {}
for i, x in enumerate(verbs):
    verbs_dict[x] = i

fname = './data/kagglecorpus/train_v2.TXT'
# test_v2.txt OR train_v2.txt
vocab, vocab_idx, idx_vocab, sents = from_text(fname, verbs)

# https://wiki.python.org/moin/UsingPickle
pickle.dump(vocab,     open(folder + 'full_vocab_1.p','wb'))
pickle.dump(vocab_idx, open(folder + 'full_vocab_idx_1.p','wb'))
pickle.dump(idx_vocab, open(folder + 'full_idx_vocab_1.p','wb')) # this one is used for the model
np.save(folder + 'full_sents_1.npy', sents)


print("Execution Time : %s seconds" % (time.time()-start_time))
################################################
