import os
import csv
import numpy as np
import gzip

from nltk import pos_tag, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
wnl = WordNetLemmatizer()

TRAIN_FILE = "../data/train.csv"
DEV_TRAIN_SPLIT = 0.1

GLOVE_VECTOR_PATH = '../data/glove/glove.6B.100d.txt.gz'

def ReadCsv(filepath, ignore_header = False):
    csvfile = open(filepath, 'rU')
    if csvfile:
        data = csv.reader((line.replace('\0','') for line in csvfile), delimiter=',',dialect=csv.excel_tab)
        if ignore_header:
            header_list = data.next()
        return data
    else:
        return False

def WriteCsv(var,filepath,Titles):
    with open(filepath, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=Titles)
        writer.writeheader()
        for row in var:
            dic = {}
            for i in range(len(row)):
                if isinstance(row[i], basestring):
                    dic[Titles[i]] = ''.join([j if ord(j) < 128 else ' ' for j in row[i]])
                else:
                    dic[Titles[i]] = str(row[i])
            writer.writerow(dic)
    return csvfile.close()

def tokenize_lemmatize_sentences(sentence_list):
    output = []
    for sentence in sentence_list:
        output.append([wnl.lemmatize(i,j[0].lower()).lower() if j[0].lower() in ['n','v'] else wnl.lemmatize(i) for i,j in pos_tag(word_tokenize(sentence))])
    return output

def pad_list(inp_list, max_len, pad_element = 0):
    n = len(inp_list)
    if n > max_len:
        return inp_list[:max_len]
    else:
        return np.array(inp_list + [pad_element]*(max_len - n))


def get_glove_mat(word_vocab, WordProcess_object):
    print("Load glove file {}\n".format(GLOVE_VECTOR_PATH))
    
    embedd_dim = -1
    with gzip.open(GLOVE_VECTOR_PATH, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue

            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1 #BECAUSE THE ZEROTH INDEX IS OCCUPIED BY THE WORD
                initW = np.random.uniform(-0.25,0.25,(len(word_vocab) + WordProcess_object.offset, embedd_dim))
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim], dtype=np.float64)
            try:
                noun_token = tokenize_lemmatize_sentneces([tokens[0]])[0][0]
                if noun_token in word_vocab:
                    index = WordProcess_object.get_key_index(noun_token)
                    if index >= WordProcess_object.offset:
                        initW[index] = tokens[1:]
            except:
                continue
    return initW
     
def pre_processing_info(data, dev_train_split = 1):
    train_set = []
    dev_set = []
    id_sentence_mapping = {}
    id_sentence_training = {}
    
    all_training_tokens = set()
    max_sent_token_len = 0
    
    count  = 0
    for row in data:
        count += 1
        if count > 1000: break
            
        row = [unicode(cell, errors = 'ignore') for cell in row]
        is_train_data = np.random.rand() > dev_train_split #### randomly spilitting train and dev data
        
        if not (row[3] and row[4]):
            continue
            
        id_sent1 = int(row[1])
        id_sent2 = int(row[2])
        
        sent1_tokens = tokenize_lemmatize_sentences([row[3]])[0]
        sent2_tokens = tokenize_lemmatize_sentences([row[4]])[0]
        id_sentence_mapping[id_sent1] = sent1_tokens
        id_sentence_mapping[id_sent2] = sent2_tokens
        
        if is_train_data:
            id_sentence_training[id_sent1] = sent1_tokens
            id_sentence_mapping[id_sent2] = sent2_tokens
            train_set.append([id_sent1, id_sent2, int(row[5])])
        else:
            dev_set.append([id_sent1, id_sent2, int(row[5])])
        
    word_tokens_lemma = id_sentence_training.values()
    for word_token in word_tokens_lemma:
        token_len = len(word_token)
        if max_sent_token_len < token_len:
            max_sent_token_len = token_len
            all_training_tokens |= set(word_token)
    print len(all_training_tokens)
    max_sent_token_len = int(0.9*max_sent_token_len)
    return train_set, dev_set, id_sentence_mapping, all_training_tokens, max_sent_token_len


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            # yield CreateVector_Object.create_train_vec(shuffled_data[start_index:end_index])
            yield shuffled_data[start_index:end_index]



class VocabProcessor():
    def __init__(self, vocab, max_len = 0, unknown = False, padding = False):
        self.padding = padding
        self.unknown = unknown
        self.offset = int(bool(unknown))
        list(vocab).sort()
        self.index_dict = {k: v+self.offset for v, k in enumerate(sorted(vocab))}
        self.max_len = max_len
        

    def get_key_index(self, key):
        # print key
        if key in self.index_dict:
            # print self.index_dict[key]
            return self.index_dict[key]
        else:
            if self.unknown:
                return self.offset - 1
            else:
                return -1
        return 0

    def get_key_indexes(self, key_list):
        output = []
        for key in key_list:
            output.append(self.get_key_index(key))
        if self.padding:
            return pad_list(output, self.max_len, 0)
        else:
            return output


class CreateVector():
    def __init__(self, word_vocab, max_sent_token_len,
                        id_sentence_mapping = {}):
        self.WordVocabProcessor_Object = VocabProcessor(word_vocab, max_sent_token_len, unknown = True, padding = True)
        self.max_sent_token_len = max_sent_token_len

        self.id_sentence_mapping = id_sentence_mapping

    def create_train_vec(self, input_trian_data):
        input_mat_sent1 = []
        input_mat_sent2 = []
        input_seq_len_sent1 = []
        input_seq_len_sent2 = []

        input_y = []
        for id_sent1, id_sent2, label in input_trian_data:
            # try:
            input_vec_sent1 = self.WordVocabProcessor_Object.get_key_indexes(self.id_sentence_mapping[id_sent1])
            input_vec_sent2 = self.WordVocabProcessor_Object.get_key_indexes(self.id_sentence_mapping[id_sent2])
            
            
            len_sent1 = min(len(self.id_sentence_mapping[id_sent1]), self.max_sent_token_len)
            len_sent2 = min(len(self.id_sentence_mapping[id_sent2]), self.max_sent_token_len)

            y = label
            # except:
            #     continue

            input_mat_sent1.append(input_vec_sent1)
            input_mat_sent2.append(input_vec_sent2)
            input_seq_len_sent1.append(len_sent1)
            input_seq_len_sent2.append(len_sent2)

            input_y.append(y)
        return zip(input_mat_sent1, input_mat_sent2, input_seq_len_sent1, input_seq_len_sent2), np.array(input_y)

    def create_sent_vector(self, sent1, sent2):
        if not sent1 or sent2:
            return False
        token_sent1, token_sent2 = tokenize_lemmatize_sentences([sent1, sent2])
        input_vec_sent1 = self.WordVocabProcessor_Object.get_key_indexes(token_sent1)
        input_vec_sent2 = self.WordVocabProcessor_Object.get_key_indexes(token_sent1)

        len_sent1 = min(len(token_sent1), self.max_sent_token_len)
        len_sent2 = min(len(token_sent1), self.max_sent_token_len)
        return input_vec_sent1, input_vec_sent2, len_sent1, len_sent2


if __name__ == "__main__":
    data = ReadCsv(TRAIN_FILE, ignore_header = True)

    print "Preprocessing Data......."
    train_set, dev_set, id_sentence_mapping, all_training_tokens, max_sent_token_len = pre_processing_info(data, 0.1)
    
    print all_training_tokens

    CreateVector_Object = CreateVector(all_training_tokens, max_sent_token_len, id_sentence_mapping)
    batches = batch_iter(train_set, 2, 1, True)
    for batch in batches:
        input_x, input_y = CreateVector_Object.create_train_vector(batch)
        input_mat_sent1, input_mat_sent2, input_seq_len_sent1, input_seq_len_sent2 = zip(*input_x)
        print np.shape(input_mat_sent1)
        print np.shape(input_mat_sent2)
        print np.shape(input_seq_len_sent1)
        print np.shape(input_seq_len_sent2)
        print np.shape(input_y)
        print input_mat_sent1[0], input_seq_len_sent1[0]
        print input_mat_sent2[0], input_seq_len_sent2[0]

        break

    print "Done"