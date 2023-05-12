import torch
import pandas as pd
from collections import Counter
import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from nltk.tokenize import word_tokenize, sent_tokenize
import re

# TOKENIZATION


class tokenization:
    def __init__(self):
        pass

    def replaceHashtags(self, txt):
        return re.sub('\#[a-zA-Z]\w+', '<HASHTAG>', txt)

    def replace_email(self, corpus):
        return re.sub(r'\S*@\S*\s?', r'<EMAIL>', corpus)

    def replaceURL(self, txt):
        return re.sub(r'(https?:\/\/|www\.)?\S+[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}\S+', r'<URL>', txt)

    def replaceMentions(self, txt):
        return re.sub(r'@\w+', r'<MENTION>', txt)

    def replaceDateTime(self, txt):
        txt = re.sub(
            r'\d{2,4}\-\d\d-\d{2,4}|\d{2,4}\/\d\d\/\d{2,4}|\d{2,4}:\d\d:?\d{2,4}', '<DATE>', txt)
        return re.sub(r'\d+:\d\d:?\d{0,2}?( am|am| pm|pm)', r'<TIME>', txt)

    # for better results as we go by string comparisions
    def upperToLower(self, txt): return txt.lower()

    def replacePunctuation(self, txt): return re.sub(
        r'(!|"|\#|\$|%|&|\'|\(|\)|\*|\+|,|-|—|’|\.|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|‘|\{|\||\}|~{1,})', r' <PUN> ', txt)

    def replaceMobileNumber(self, txt):
        return re.sub(r'[\+0-9\-\(\)\.]{3,}[\-\.]?[0-9\-\.]{3,}', r'<MOB>', txt)

    def replaceNumericals(self, txt):
        # returning the numericals as <NUM> as we are not considering the numericals in the corpus
        return re.sub(r'\d+', r'<NUM>', txt)


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
    return corpus.split()


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
    ):
        self.args = args
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()
        # to get word from index
        self.index_to_word = {index: word for index,
                              word in enumerate(self.uniq_words)}
        # to get index from word
        self.word_to_index = {word: index for index,
                              word in enumerate(self.uniq_words)}

        # getting indexes of words in the dataset
        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        with open('train.txt', 'r') as f:
            text = f.read()
        return text.split(' ')

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    # len function used to get the length of the dataset so that it suits the dataloader at the last step
    def __len__(self):
        return len(self.words_indexes) - self.args.sequence_length

    # calling the dataset object with an index will return the corresponding item
    def __getitem__(self, index):
        return (
            torch.tensor(
                self.words_indexes[index:index+self.args.sequence_length]),
            torch.tensor(
                self.words_indexes[index+1:index+self.args.sequence_length+1]),
        )


class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.input_size = 128
        self.hidden_state_size = 128
        self.embedding_dim = 128
        self.num_lstm_layers = 1
        # getting the no. of unique words in the dataset
        n_vocab = len(dataset.uniq_words)

        # Emnbedding layer
        # DOUBT IF WE DEFINE EMBEDDING LAYER THEN HOW DO WE GET THE EMBEDDING OF UNKNOWNS
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_state_size,
            num_layers=self.num_lstm_layers,
        )
        # Fully connected layer -> output layer -> get logits that are unnormalized log probabilities
        # for each word in the sequence
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_state_size, 256),
            nn.Linear(256, n_vocab)
        )
    # forward method

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    # initiator for the hidden state and cell state
    def init_state(self, sequence_length):
        return (torch.zeros(self.num_lstm_layers, sequence_length, self.hidden_state_size),
                torch.zeros(self.num_lstm_layers, sequence_length, self.hidden_state_size))

def train(dataset, model, args):
    model.train()
    # the dataloader that will iterate through the dataset and return batches
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    # print(args.batch_size)
    # the loss function => CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    # the optimizer that goes through the model and updates the weights then update it with 0.01 learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # iterating through the epochs
    for epoch in range(args.max_epochs):
        state_h, state_c = model.init_state(args.sequence_length)
        # iterating through the batches in each epoch
        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()   # sets the gradients to 0
            # update the hidden state and cell state
            y_pred, (state_h, state_c) = model(x, (state_h, state_c))

            print(y_pred.shape, y.shape)
            # calculate the loss
            loss = criterion(y_pred.transpose(1, 2), y)

            # hidden state and cell state are detached from the graph
            state_h = state_h.detach()
            state_c = state_c.detach()

            # backpropagation
            loss.backward()
            # update the weights
            optimizer.step()
            # print the loss
            
        print({'epoch': epoch, 'loss': loss.item()})


def predict(dataset, model, text):
    model.eval()
    our_word = text[-1]
    text = text[:-1]
    our_word_index = dataset.word_to_index[our_word]
    state_h, state_c = model.init_state(3)
    print(text)
    x = torch.tensor([[dataset.word_to_index[w]
                     for w in text[i:]]]).to(torch.int64)
    # torch.reshape(x, (0,1))
    print(x.shape)
    y_pred, (state_h, state_c) = model(x, (state_h, state_c))
    last_word_logits = y_pred[0]
    p = torch.nn.functional.softmax(
        last_word_logits, dim=0).detach().numpy()
    # print(p.shape)
    our_word_prob = p[0, our_word_index]
    return our_word_prob


def calculate_perplexity(dataset, model, sentence):
    # pass 4 grams one by one from the sentence to predict fucntion and get the probability of the last word
    # then calculate the perplexity
    perplexity = 1
    for i in range(len(sentence)-3):
        print(sentence)
        perplexity *= predict(dataset, model, sentence[i:i+6])
    perplexity = perplexity ** (-1/len(sentence))
    return perplexity

def test(model, test_dataset, args):
    model.eval()
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    print((test_dataloader))
    with torch.no_grad():
        for x, y in (test_dataloader):
            state_h, state_c = model.init_state(1)
            state_h = state_h
            state_c = state_c
            x = x
            y = y
            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            print(y_pred.shape,y.shape)
            loss = criterion(y_pred.transpose(1, 2), y)
            total_loss += loss.item()

    test_loss = total_loss / len(test_dataloader)
    return test_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--sequence-length', type=int, default=8)
    args = parser.parse_args()
    list1_of_sentences = []
    with open("./test.txt", "r") as f:
        data = f.read()
    #=======================cleaning the data========================#
    newlines = [".", "?", "!"]
    corpus1 = data.replace("\n", " ")
    for ele in newlines:
        corpus1 = corpus1.replace(ele, ele+"\n")

    for (sentence) in corpus1.splitlines():
        cleaned_list = tokenize(sentence)
        if(len(cleaned_list) != 0):
            cleaned_sent = " ".join(_ for _ in cleaned_list)
            list1_of_sentences.append(cleaned_sent)

    #=======================splitting the data into train and test set========================#
    for_test = int(0.3*len(list1_of_sentences))
    np.random.seed(11)
    idx = np.random.choice(len(list1_of_sentences), for_test, replace=False)
    trainset = []
    testset = []

    for id in range(len(list1_of_sentences)):
        if id in idx:
            testset.append(list1_of_sentences[id])
        else:
            trainset.append(list1_of_sentences[id])

    # adding the testset to the test.txt file
    with open("test.txt", "w") as f:
        f.write(" ".join((_) for _ in testset))

    # replace the words that occuer only once with <unk> in the trainset set
    vocab = {}
    for i in range(len(trainset)):
        trainset[i] = trainset[i].split()
        for j in range(len(trainset[i])):
            if trainset[i][j] in vocab:
                vocab[trainset[i][j]] += 1
            else:
                vocab[trainset[i][j]] = 1
    for i in range(len(trainset)):
        for j in range(len(trainset[i])):
            if vocab[trainset[i][j]] == 1:
                trainset[i][j] = "<unk>"

    trainset = " ".join([" ".join(_) for _ in trainset])

    with open("train.txt", "w") as f:
        f.write(trainset)

    
    # intialize the dataset and the model
    dataset = Dataset(args)
    model = Model(dataset)

    # train the model
    train(dataset, model, args)
    #=====================SAVING MODEL==========================#
    torch.save(model.state_dict(), "model1.pth")
    # sentences = ["what a fine thing"]
    
    # # get perplexity for the given sentence
    # for sentence in sentences:
    #     words = sentence.split()
    #     for i in range(len(words)):
    #         if words[i] not in dataset.uniq_words:
    #             words[i] = "<unk>"
    #     # join the words to get the sentence
    #     perplexity = calculate_perplexity(dataset, model, words)

    #=====================TESTING==========================#
    # test(model,testset,args)
