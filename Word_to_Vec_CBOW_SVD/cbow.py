# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1E6qbEeH-xfpN5khexIYPWDRtj7rIskZ-
"""

import sys
import json
import re
import torch
import collections
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import collections
import torch.optim as optim
import torch.utils.data
import time
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")

class tokenization:
    def __init__(self):
        pass

    def replaceHashtags(self, txt):
        return re.sub('\#[a-zA-Z]\w+', ' ', txt)

    def replace_email(self, corpus):
        return re.sub(r'\S*@\S*\s?', r' ', corpus)

    def replaceURL(self, txt):
        return re.sub(r'(https?:\/\/|www\.)?\S+[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}\S+', r' ', txt)

    def replaceMentions(self, txt):
        return re.sub(r'@\w+', r' ', txt)

    def replaceDateTime(self, txt):
        txt = re.sub(
            r'\d{2,4}\-\d\d-\d{2,4}|\d{2,4}\/\d\d\/\d{2,4}|\d{2,4}:\d\d:?\d{2,4}', '', txt)
        return re.sub(r'\d+:\d\d:?\d{0,2}?( am|am| pm|pm)', r' ', txt)

    # for better results as we go by string comparisions
    def upperToLower(self, txt): return txt.lower()


    def replacePunctuation(self, txt): return re.sub(
        r'(!|"|\#|\$|%|&|\'|\(|\)|\*|\+|,|-|—|’|\.|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|‘|\{|\||\}|~{1,})', r' ', txt)

    def replaceMobileNumber(self, txt):
        return re.sub(r'[\+0-9\-\(\)\.]{3,}[\-\.]?[0-9\-\.]{3,}', r' ', txt)

    def replaceNumericals(self, txt):
        # returning the numericals as <NUM> as we are not considering the numericals in the corpus
        return re.sub(r'\d+', r' ', txt)

def tokenize(corpus):
    token = tokenization()
    corpus = token.replaceHashtags(corpus)
    corpus = token.replace_email(corpus)
    corpus = token.replaceURL(corpus)
    corpus = token.replaceMentions(corpus)
    corpus = token.upperToLower(corpus)
    corpus = token.replaceDateTime(corpus)
    corpus = token.replacePunctuation(corpus)
    corpus = token.replaceMobileNumber(corpus)
    corpus = token.replaceNumericals(corpus)
    return corpus

class Corpus:
    def __init__(self, data,minimum_freq = 0, sub = 1):
        self.sub = sub
        self.corpus = self.read_text_line(data)
        self.tokenized_corpus = self.tokenize_corpus(self.corpus)
        self.data, self.count, self.dictionary, self.reverse_dictionary = self.build_dataset_line(self.tokenized_corpus, minimum_freq)
        self.vocab_size = len(self.dictionary) - 1
        self.negaive_sample_table_w, self.negaive_sample_table_p = self.create_negative_sample_table()


    def read_text_line(self, file):
        sentences = []
        sentences_tokenized = []
        with open(file, 'r') as datafile:
            index = 0
            for line in datafile:
                review = json.loads(line)
                line = line.split(""""reviewText":""")[1]
                line = line.split(""", "overall":""")[0]
                # get sentences from each review
                line = re.split(r' *[\.\?!][\'"\)\]]* *', line)
                for sentence in line:
                    if(len(sentence) > 0):
                        sentences.append(sentence)
                        if(len(tokenize(sentence))> 0):
                          sentences_tokenized.append(tokenize(sentence))
                        if(len(sentences_tokenized)>40000):
                          index = 1
                          break
                if(index == 1):
                  break
        return sentences_tokenized

    
    def build_dataset_line(self, tokenized_corpus, minimum_freq):
        count = [['UNK', -1]]
        words = np.concatenate(tokenized_corpus)
        
        count.extend(collections.Counter(words).most_common())
        count = np.array(count)
        count = count[(count[:, 1].astype(int) >= minimum_freq) | (count[:, 1].astype(int) == -1)]
        # unable to do sub-sampling

        #dictionary
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)

        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        #data
        data = list()
        unk_count = 0
        for r in tokenized_corpus:
            # print(r)
            data_tmp = list()
            for word in r:
                if word in dictionary:
                    index = dictionary[word]
                else:
                    index = 0  # dictionary['UNK']
                    unk_count += 1
                data_tmp.append(index)
            if(len(data_tmp) > 0):
                data.append(data_tmp)
        count[0][1] = unk_count
        return data, count, dictionary, reverse_dictionary

    def tokenize_corpus(self, corpus):
        tokens = [np.array(x.split()) for x in corpus]
        return tokens
    
    def create_negative_sample_table(self):
        word_counts = self.count[1:, 1].astype(int) #index 0 is unk
        p = word_counts / word_counts.sum()
        p = np.power(p, 0.75)
        p /= p.sum()
        
        return np.array(list(range(1, len(self.dictionary)))), p

class word2vec(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, negative_samples, ns):
        super(word2vec, self).__init__()
        self.ns = ns
        self.negative_samples = negative_samples
        self.row_idx = 0
        self.col_idx = 0
        self.batch_end = 0
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size + 1        
        self.u_embeddings = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.v_embeddings = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        if torch.cuda.is_available():
            self.u_embeddings = self.u_embeddings.cuda()
            self.v_embeddings = self.v_embeddings.cuda()

        #embedding init
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

# main function
if __name__ == "__main__":
    start = time.time()
    corpus = Corpus(data = 'test.json',
                minimum_freq = 0)

    window_size = 4
    embedding_dims = 300
    batch_size = 256