from .imports import *
# define dataset classlike
class Dataset(torch.utils.data.Dataset):
    """
    Dataset class
    TODO:
        * Rather than just passing the index of the word, pass the embedding 
        of the word from the ./embeddings.txt file.
    """
    def __init__(self, data,config):
        self.data = data
        self.word_to_ix = config['word_to_ix']
        self.embedding_dim = config['embedding_dim']
        self.seq_len = config['seq_len']
        self.device = config['device']

    def __getitem__(self, index):
        sample = list(self.data.values())[index]
        sentence_input_sample = []
        sentence_target_sample = []
        for info in sample:
            input_batch = []
            target_batch = []
            words = info[0].split()
            i = 0
            for word in words:
                if(i<self.seq_len):
                    if(word not in self.word_to_ix.keys()):
                        input_batch.append(self.word_to_ix["unk"])
                        target_batch.append((self.word_to_ix["unk"]))
                    else:
                        input_batch.append((self.word_to_ix[word]))
                        target_batch.append((self.word_to_ix[word]))
                i += 1
            # input_batch = input_batch[:-1]
            # convert input_batch list to tensor
            # target_batch = target_batch[1:]
            # pad front and end of the target batch
            # if the sentence is less than the sequence length, pad it
            if(len(input_batch) < self.seq_len):
                input_batch = input_batch +  [(self.word_to_ix["pad"]) for i in range(self.seq_len - len(input_batch))]
                target_batch = target_batch + [(self.word_to_ix["pad"]) for i in range(self.seq_len - len(target_batch))] 
            target_batch = [self.word_to_ix["sos"]] + target_batch + [self.word_to_ix["eos"]]
            # stack the input and target batch
            input_batch = torch.tensor(input_batch).to(self.device)
            target_batch = torch.tensor(target_batch).to(self.device)
            sentence_input_sample.append(input_batch)
            sentence_target_sample.append(target_batch)
        # remove the last element from the input batch tensor
        input_batch = torch.stack(sentence_input_sample).to(self.device)
        target_batch = torch.stack(sentence_target_sample).to(self.device)
        return input_batch, target_batch

    def __len__(self):
        return len(self.data)

# define DataLoader
class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, num_workers=8):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def __iter__(self,shuffle=False):
        if self.shuffle:
            np.random.shuffle(self.dataset.data)
        self.batch_num = 0
        return self

    def __next__(self):
        if self.batch_num * self.batch_size >= len(self.dataset):
            raise StopIteration
        batch = self.dataset[self.batch_num * self.batch_size: (self.batch_num + 1) * self.batch_size]
        self.batch_num += 1
        return batch
