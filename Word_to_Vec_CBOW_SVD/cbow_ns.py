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

    def generate_batch(self, corpus, window_size):
        row_idx = self.row_idx
        col_idx = self.col_idx
        context = collections.deque()
        target  = collections.deque()
        i = 0

        while row_idx < len(corpus.data):

            data = corpus.data[row_idx]
            target_ = data[col_idx]
            sentence_length = len(data)

            start_idx = col_idx - window_size
            start_idx = 0 if  start_idx < 0 else start_idx
            end_idx = col_idx + 1 + window_size
            end_idx = end_idx if  end_idx < (sentence_length )  else sentence_length

            c = [data[x] for i, x in enumerate(range(start_idx, end_idx)) if x != col_idx]
            if len(c) == (window_size * 2):
                context.append(c)
                target.append(target_)
            i += 1
            
            col_idx = (col_idx + 1)
            if col_idx == len(data):
                col_idx  = 0
                row_idx = row_idx + 1

        x = np.array(context)
        y = np.array(target)
        return x, y

    def negative_sampling(self, corpus):
        negative_samples = np.random.randint(low = 1, high = self.vocab_size, size = self.negative_samples)
        return negative_samples    

    def forward(self, batch, corpus = None):
        if self.ns == 0:
            u_emb = torch.mean(self.u_embeddings(batch[0]), dim = 1)
            v_emb = self.v_embeddings(torch.LongTensor(range(self.vocab_size)))
            z = torch.matmul(u_emb, torch.t(v_emb))

            log_softmax = F.log_softmax(z, dim = 1)
            loss = F.nll_loss(log_softmax, batch[1])
        else:
            #positive
            u_emb = torch.mean(self.u_embeddings(batch[0]), dim = 1)
            v_emb = self.v_embeddings(batch[1])

            score = torch.sum(torch.mul(u_emb, v_emb), dim = 1)#inner product
            log_target = F.logsigmoid(score)

            #negative
            v_emb_negative = self.v_embeddings(batch[2])
            neg_score = -1 * torch.sum(torch.mul(u_emb.view(batch[0].shape[0], 1, self.embedding_dim), v_emb_negative.view(batch[0].shape[0], batch[2].shape[1], self.embedding_dim)), dim = 2)
            log_neg_sample = F.logsigmoid(neg_score)

            loss = -1 * (log_target.sum() + log_neg_sample.sum())
        return loss

class Use_model:
    def __init__(self, corpus, embedding_dim, window_size, batch_size, 
                 negative_samples = 10, ns = 0, trace = False):
        self.corpus = corpus
        self.window_size = window_size
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.ns = ns
        self.negative_samples = negative_samples
        self.trace = trace
        self.model  = word2vec(self.corpus.vocab_size, self.embedding_dim, self.negative_samples, self.ns)
        
    def train(self, num_epochs = 100, learning_rate = 0.001):
        optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)
        
        x, y = self.model.generate_batch(self.corpus, self.window_size)
        x = torch.LongTensor(x).to(device)
        y = torch.LongTensor(y).to(device)
        epo_start_time = time.time()
        for epo in range(num_epochs):
            loss_val = 0
            if self.ns == 1:
                ns = torch.LongTensor(np.array([self.model.negative_sampling(self.corpus) for i in range(len(x))])).to(device)
            if self.ns == 0:
                dataset = torch.utils.data.TensorDataset(x, y)
            else:
                dataset = torch.utils.data.TensorDataset(x, y, ns)

            batches = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

            for batch in batches:
                optimizer.zero_grad()
                loss = self.model(batch)
                loss.backward()
                loss_val += loss.data
                optimizer.step()
            if(epo%2 == 0):
                print(time.time() - epo_start_time)
                epo_start_time = time.time()
                print(f'Loss at epo {epo}: {loss_val/len(batches)}')
            
        # save model            
        torch.save(self.model.state_dict(), f"model_{self.embedding_dim}_{self.batch_size}_{num_epochs}.pt") 
    def get_vector(self, word):
        word_idx = self.corpus.dictionary[word]
        word_idx = torch.LongTensor([[word_idx]])
        if torch.cuda.is_available():
            word_idx = word_idx.to(device)
            vector = self.model.u_embeddings(word_idx).view(-1).detach().cpu().numpy()
        else:
            vector = self.model.u_embeddings(word_idx).view(-1).detach().numpy()
        return vector

    def similarity_pair(self, word1, word2):
        return np.dot(self.get_vector(word1), self.get_vector(word2)) / (np.linalg.norm(self.get_vector(word1)) * np.linalg.norm(self.get_vector(word2)))
    
    def similarity(self, word, descending = True):
        words = np.array([x for x in self.corpus.dictionary.items()])
        
        sim = np.array(list(map(lambda x: self.similarity_pair(word, x[0]), words)))#calculate similarity
        sim_list = np.vstack((sim, words[:, 0])).T
        
        if descending:
            rnk = np.argsort(sim, )[::-1]
        else:
            rnk = np.argsort(sim, )
    
        sim_list[rnk]
        return sim_list[rnk]

# main function
if __name__ == "__main__":
    start = time.time()
    corpus = Corpus(data = 'test.json',
                minimum_freq = 0)

    window_size = 4
    embedding_dims = 300
    batch_size = 256
    model = Use_model(corpus, embedding_dims, window_size, batch_size, 
                                    ns = 1, negative_samples = 5, trace = True)
    model.train(num_epochs = 20, learning_rate = 0.01)
    process_time = time.time() - start
    print("Total_Training_Time: ", process_time)
    words_similar = model.similarity("love")
    print(words_similar[0:10])

"""**Adam OPTIMIZER, LR = 0.01, EPOCHS = 20, EMBEDDING_DIM = 300, BATCH_SIZE = 256 window_size= 4 neg_samples = 5(default = 5)**"""

words_similar = model.similarity("titanic")
print(words_similar[0:20])

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
tsne = TSNE(n_components=2, random_state=0)
words = np.array([x for x in corpus.dictionary.items()])
# words = words[0:1000]
vectors = np.array(list(map(lambda x: model.get_vector(x[0]), words)))
Y = tsne.fit_transform(vectors)
labels = words[:, 0]

plt.figure(figsize=(16, 9))
words_similar = model.similarity("movie")
# plot the embeddings concentrating on the 10 nearest neighbors of the word and don't need to show all words in the plot
for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
    if label == "movie":
        plt.scatter(x, y, c='r')
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    elif label in words_similar[0:10,1]:
        plt.scatter(x, y, c='b')
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

plt.figure(figsize=(16, 9))
words_similar = model.similarity("award")
# plot the embeddings concentrating on the 10 nearest neighbors of the word and don't need to show all words in the plot
for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
    if label == "award":
        plt.scatter(x, y, c='r')
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    elif label in words_similar[0:10,1]:
        plt.scatter(x, y, c='b')
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

plt.figure(figsize=(16, 9))
words_similar = model.similarity("good")
# plot the embeddings concentrating on the 10 nearest neighbors of the word and don't need to show all words in the plot
for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
    if label == "good":
        plt.scatter(x, y, c='r')
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    elif label in words_similar[0:10,1]:
        plt.scatter(x, y, c='b')
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

# code to print all embeddings into line word followed by its embedding
with open('embeddings.txt', 'w') as f:
    for word in corpus.dictionary:
        f.write(word + ' ' + ' '.join(map(str, model.get_vector(word))) + ' '  )