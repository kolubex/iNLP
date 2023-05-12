import os
import io
import conllu
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from datetime import datetime
# print = lambda *n: ...

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a function to extract the required columns from the dataset


def extract_columns(data):
    result = []
    for sentence in data:
        sent_tokens = []
        for token in enumerate(sentence):
            sent_tokens.append((token[0], token[1]["form"], token[1]["upos"]))
        result.append(sent_tokens)
    return result


def get_words_and_tags(data, embedding_dim):
    all_words = []
    for sentence in data:
        for token in sentence:
            all_words.append(token[1])

    # add all tags into a list
    all_tags = []
    for sentence in data:
        for token in sentence:
            all_tags.append(token[2])
    # create tag_index dictionary
    tag_to_ix = {}
    for tag in all_tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
    tag_to_ix["<UNK>"] = len(tag_to_ix)

    # print(tag_to_ix)
    word_to_ix = {}
    for word in all_words:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    word_to_ix["<UNK>"] = len(word_to_ix)

    # replace token[1] with its embedding for each token in the sentence
    final_data = []
    # print(data)
    for sentence in (data):
        # print(sentence)
        sentence_temp = []
        for token in sentence:
            token = (word_to_ix[token[1]], tag_to_ix[token[2]])
            sentence_temp.append(token)
        final_data.append(sentence_temp)
    # print(final_data)
    return final_data, word_to_ix, tag_to_ix

# ACTUAL MODEL
# defining class for LSTM model


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, num_layers):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size).to(device)

    def forward(self, sentence):
        # for i in range(len(sentence)):
        #     sentence[i] = sentence[i].to(device)
        #     embeds.app
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(
            embeds.view(len(sentence), 1, -1))
        lstm_out = lstm_out.to(device)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

# Validating model with validation data


def get_accuracy(model, validating_data):
    with torch.no_grad():
        correct = 0
        total = 0
        for data in validating_data:
            inputs = []
            labels = []
            for ele in data:
                inputs.append(ele[0])
                labels.append(ele[1])
            inputs = torch.tensor(inputs).to(device)
            labels = torch.tensor(labels).to(device)
            outputs = model(inputs)
            # print outputs from valid_tag_to_ix to get the actual tags
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the %d validation examples: %d %%' %
              (total, 100 * correct / total))
        return (100 * correct / total)


def train_model(model, training_data, loss_function, optimizer, EPOCHS, BATCH_SIZE, validating_data, train_tag_to_ix):
    running_loss = 0.0
    batch_loss = 0.0
    # TRAINING THE MODEL
    model.train()
    prev_accuracy = 0
    with torch.cuda.device(device):
        for epoch in range(0, EPOCHS):
            running_loss = 0.0
            print("Starting", {epoch}, "Epoch", datetime.now())
            # taking batches by slicing
            for i in range(0, len(training_data), BATCH_SIZE):
                mini_batch = training_data[i:i+BATCH_SIZE]
                batch_loss = 0.0
                model.zero_grad()
                for (data) in (mini_batch):
                    inputs = []
                    labels = []
                    for ele in data:
                        inputs.append(ele[0])
                        labels.append(ele[1])
                    inputs = torch.tensor(inputs).to(device)
                    labels = torch.tensor(labels).to(device)

                    outputs = model((inputs))

                    loss = loss_function(outputs, labels)
                    batch_loss += loss
                    running_loss += loss
                batch_loss = batch_loss/BATCH_SIZE
                batch_loss.backward(retain_graph=True)
                optimizer.step()

            print(f"{epoch-1}_Epoch Accuracy - {prev_accuracy}")
            prev_accuracy = get_accuracy(model, validating_data)
            print(f"{epoch}_Epoch Accuracy - {prev_accuracy}")
            print(
                f"Epoch {epoch} completed with loss {running_loss/len(training_data)}")

# Saving the model


def save_model(model, EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE, EPOCHS, NUM_LAYERS):
    torch.save(model.state_dict(
    ), f"model_{EMBEDDING_DIM}_{HIDDEN_DIM}_{BATCH_SIZE}_{EPOCHS}_{NUM_LAYERS}.pt")

# Testing Model


def test_model(model, testing_data, train_tag_to_ix):
    with torch.no_grad():
        correct = 0
        total = 0
        all_labels = []
        all_predicted = []
        for data in testing_data:
            inputs = []
            labels = []
            for ele in data:
                inputs.append(ele[0])
                labels.append(ele[1])
            inputs = torch.tensor(inputs).to(device)
            labels = torch.tensor(labels).to(device)
            outputs = model((inputs))
            outputs = model(inputs)
            # print outputs from valid_tag_to_ix to get the actual tags
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            labels = (labels.cpu().numpy())
            predicted = (predicted.cpu().numpy())
            for i in labels:
                all_labels.append(i)
            for i in predicted:
                all_predicted.append(i)
        print('Accuracy of the network on the %d test examples: %d %%' %
              (total, 100 * correct / total))
        # get precision, recall and f1 score
        print(classification_report(all_labels, all_predicted,
              target_names=list(train_tag_to_ix.keys()).remove("<UNK>")))
        print(confusion_matrix(all_labels, all_predicted))


# MAIN FUNCTION
if __name__ == "__main__":
    # if model_300_256_4_2_3.pt file is in the same folder then load the model
    # else train the model
    EMBEDDING_DIM = 300
    BATCH_SIZE = 4
    HIDDEN_DIM = 256
    EPOCHS = 2
    NUM_LAYERS = 3
    path = "./UD_English-Atis"
    train_file = os.path.join(path, "en_atis-ud-train.conllu")
    with io.open(train_file, "r", encoding="utf-8") as f:
        train_data = conllu.parse(f.read())
        train_data = extract_columns(train_data)
    training_data, train_word_to_ix, train_tag_to_ix = get_words_and_tags(
        train_data, EMBEDDING_DIM)
    if os.path.isfile("model_300_256_4_2_3.pt"):
        model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(
            train_word_to_ix), len(train_tag_to_ix), NUM_LAYERS)
        model.load_state_dict(torch.load("model_300_256_4_2_3.pt"))
        model = model.to(device)
        print("MODEL LOADED!")
        model.eval()
        while True:
            sentence = input("Enter your sentence: ")
            temp = sentence
            sentence = sentence.split()
            temp = temp.split()
            sentence = [(train_word_to_ix[word] if word in train_word_to_ix.keys(
            ) else train_word_to_ix["<UNK>"]) for word in sentence]
            sentence = torch.tensor(sentence).to(device)
            output = model(sentence)
            _, predicted = torch.max(output.data, 1)
            predicted = (predicted.cpu().numpy())
            for num,i in enumerate(predicted):
                print(temp[num],list(train_tag_to_ix.keys())[i])
    else:
        # PARMAETERS
        # LEARNING_RATE = 0.1
        # Set the path to the directory containing the dataset files
        path = "./UD_English-Atis"
        # Load the  dev, and test files
        train_file = os.path.join(path, "en_atis-ud-train.conllu")
        dev_file = os.path.join(path, "en_atis-ud-dev.conllu")
        test_file = os.path.join(path, "en_atis-ud-test.conllu")

        # Read the files using the conllu library and extract the required columns
        with io.open(train_file, "r", encoding="utf-8") as f:
            train_data = conllu.parse(f.read())
            train_data = extract_columns(train_data)

        with io.open(dev_file, "r", encoding="utf-8") as f:
            dev_data = conllu.parse(f.read())
            dev_data = extract_columns(dev_data)

        with io.open(test_file, "r", encoding="utf-8") as f:
            test_data = conllu.parse(f.read())
            test_data = extract_columns(test_data)

        training_data, train_word_to_ix, train_tag_to_ix = get_words_and_tags(
            train_data, EMBEDDING_DIM)

        # Getting validating_data, testing_data.

        validating_data = []
        for sentence in dev_data:
            temp_sentence = []
            for word in sentence:
                if word[1] in train_word_to_ix.keys():
                    index = train_word_to_ix[word[1]]
                    if word[2] in train_tag_to_ix.keys():
                        index2 = train_tag_to_ix[word[2]]
                        temp_sentence.append((index, index2))
                    else:
                        index2 = train_tag_to_ix["<UNK>"]
                        temp_sentence.append((index, index2))
                else:
                    index = train_word_to_ix["<UNK>"]
                    if word[2] in train_tag_to_ix.keys():
                        index2 = train_tag_to_ix[word[2]]
                        temp_sentence.append((index, index2))
                    else:
                        index2 = train_tag_to_ix["<UNK>"]
                        temp_sentence.append((index, index2))
            validating_data.append(temp_sentence)

        # Doing the same on test data to get testing_data
        testing_data = []
        for sentence in test_data:
            temp_sentence = []
            for word in sentence:
                if word[1] in train_word_to_ix.keys():
                    index = train_word_to_ix[word[1]]
                    if word[2] in train_tag_to_ix.keys():
                        index2 = train_tag_to_ix[word[2]]
                        temp_sentence.append((index, index2))
                    else:
                        index2 = train_tag_to_ix["<UNK>"]
                        temp_sentence.append((index, index2))
                else:
                    index = train_word_to_ix["<UNK>"]
                    if word[2] in train_tag_to_ix.keys():
                        index2 = train_tag_to_ix[word[2]]
                        temp_sentence.append((index, index2))
                    else:
                        index2 = train_tag_to_ix["<UNK>"]
                        temp_sentence.append((index, index2))
            testing_data.append(temp_sentence)

        model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM,  len(
            train_word_to_ix), len(train_tag_to_ix), NUM_LAYERS)
        model = model.to(device)
        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters())
        train_model(model, training_data, loss_function, optimizer,
                    EPOCHS, BATCH_SIZE, validating_data, train_tag_to_ix)
        save_model(model, EMBEDDING_DIM, HIDDEN_DIM,
                   BATCH_SIZE, EPOCHS, NUM_LAYERS)
        get_accuracy(model, validating_data)
        test_model(model, testing_data, train_tag_to_ix)
        print("TRAINING AND TESTING DONE!")
        print("PRESS T TO TEST YOUR SENTENCE")
        print("PRESS Q TO QUIT")
        while True:
            input1 = input("Enter your choice: ")
            if input1 == "T":
                sentence = input("Enter your sentence: ")
                sentence = sentence.split()
                sentence = [(train_word_to_ix[word] if word in train_word_to_ix.keys(
                ) else train_word_to_ix["<UNK>"]) for word in sentence]
                sentence = torch.tensor(sentence).to(device)
                output = model(sentence)
                _, predicted = torch.max(output.data, 1)
                predicted = (predicted.cpu().numpy())
                for i in predicted:
                    print(list(train_tag_to_ix.keys())[i])
            else:
                break
