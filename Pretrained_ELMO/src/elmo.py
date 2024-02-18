from utils.imports import *
from utils.data_elmo import Dataset, DataLoader
from utils.preprocessing import preprocess, create_vocab

class LSTM(nn.Module):
     # give me a lstm model that is used for classification of a sample
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['vocab_size']
        self.n_layers = config['num_layers']
        self.bidirectional = config['bidirectional']
        self.batch_first = config['batch_first']
        self.embedding_layer = nn.Embedding(self.output_dim, self.embedding_dim)
        self.lstm1_forward = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=self.batch_first)
        self.lstm1_backward = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=self.batch_first)
        self.lstm2_forward = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=self.batch_first)
        self.lstm2_backward = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=self.batch_first)
        self.fc_forward = nn.Linear(self.hidden_dim, self.output_dim)
        self.fc_backward = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = self.embedding_layer(x)
        forward_output1, (hidden, cell) = self.lstm1_forward(x)
        # reverse the tensor along the last dimension
        x = torch.flip(x, [2])
        backward_output1, (hidden, cell) = self.lstm1_backward(x)
        # forward this outpt to the second lstm layer
        forward_output2, (hidden, cell) = self.lstm2_forward(forward_output1)
        backward_output2, (hidden, cell) = self.lstm2_backward(backward_output1)
        # take mean of forward_output1 and forward_output2. Do the same for backward_output1 and backward_output2
        forward_output =self.fc_forward(forward_output2)
        backward_output = self.fc_backward(backward_output2)
        
        return {
            "word_embeddings": x,
            "layer1_output_forward": forward_output1,
            "layer1_output_backward": backward_output1,
            "layer2_output_forward": forward_output2,
            "layer2_output_backward": backward_output2,
            "forward_prediction": forward_output,
            "backward_prediction": backward_output
        }

@torch.no_grad()
def get_loss(model, config, dataloader):
    """
    Args:
        model: the model to train
        dataloader: the dataloader to get the data
        config: configuration of the model
    Returns:
        loss: the loss of the model
    """
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0
    model.eval()
    batch_num = 0
    # get batches from the dataloader
    for batch in dataloader:
            # get the input and target batch
            input_batch = batch[0]
            target_batch = batch[1]
            outputs = model(input_batch)
            foward_preds = outputs["forward_prediction"]
            backward_preds = outputs["backward_prediction"]
            forward_target = target_batch[:,2:]
            backward_target = target_batch[:,:-2]
            # print all above 4 vectors shapes
            foward_preds = foward_preds.permute(0, 2, 1)
            backward_preds = backward_preds.permute(0, 2, 1)
            loss = (loss_function(foward_preds, forward_target) + loss_function(backward_preds, backward_target))/2
            # add the loss to the total loss
            total_loss += loss.item()
            batch_num += 1
    return total_loss/batch_num

def train_nn(model, config, train_dataloader,val_dataloader):
    """
    Args:
        model: the model to train
        train_data: the training data
        epochs: number of epochs
        lr: learning rate
        batch_size: size of the batch
        config: configuration of the model
    Returns:
        model: the trained model
    """
    epochs = config['epochs']
    lr = config['lr']
    loss_function = nn.CrossEntropyLoss()
    if config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5,factor=0.1)
    best_val_loss = 1e9
    best_model = None
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        batch_num = 0
        # get batches from the dataloader
        for batch in train_dataloader:
            # get the input and target batch
            input_batch = batch[0]
            target_batch = batch[1]
            # zero the gradients
            model.zero_grad()
            # get the output of the model
            outputs = model(input_batch)
            foward_preds = outputs["forward_prediction"]
            backward_preds = outputs["backward_prediction"]
            forward_target = target_batch[:,2:]
            backward_target = target_batch[:,:-2]
            # print all above 4 vectors shapes
            foward_preds = foward_preds.permute(0, 2, 1)
            backward_preds = backward_preds.permute(0, 2, 1)
            loss = (loss_function(foward_preds, forward_target) + loss_function(backward_preds, backward_target))/2
            # backpropagate the loss
            loss.backward()
            # update the parameters
            optimizer.step()
            # add the loss to the total loss
            total_loss += loss.item()
            batch_num += 1
        print("Epoch: {}, Loss: {}".format(epoch, total_loss/batch_num))
        val_loss = get_loss(model, config, val_dataloader)
        print(f"Loss on the val set: {val_loss}")
        train_loss = total_loss/batch_num
        data_to_log = {
            "lr": optimizer.param_groups[0]['lr'],
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        print(data_to_log)
        wandb.log(data_to_log)
        if not os.path.exists("/ssd_scratch/cvit/kolubex_anlp_elmo/"):
            os.makedirs("/ssd_scratch/cvit/kolubex_anlp_elmo/")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # save the model
            torch.save(model.state_dict(), f"/ssd_scratch/cvit/kolubex_anlp_elmo/{config['model_name']}.pth")
            best_model = copy.deepcopy(model)
            # save the copy of the model in best_model
            print("Model saved!")
    return best_model

def get_embeddings(model, config, dataloader):
    """
    Args:
        model: the model to train
        dataloader: the dataloader to get the data
        config: configuration of the model
    Returns:
        loss: the loss of the model
    """
    outputs_list = []
    total_loss = 0
    model.eval()
    batch_num = 0
    # get batches from the dataloader
    for batch in dataloader:
            # get the input and target batch
            input_batch = batch[0]
            target_batch = batch[1]
            outputs = model(input_batch)
            outputs_list.append(outputs)
    return outputs_list

# load the model and get the embeddings.
class ELMO_loaded(nn.Module):
    def __init__(self, config):
        super(ELMO_loaded, self).__init__()
        self.config = config
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['vocab_size']
        self.n_layers = config['num_layers']
        self.bidirectional = config['bidirectional']
        self.batch_first = config['batch_first']
        self.model = self.get_model(config)
    
    @torch.no_grad()
    def get_model(self,config):
        model = LSTM(config)
        state_dict = torch.load(f"/ssd_scratch/cvit/kolubex_anlp_elmo/{config['model_name']}.pth")
        model.load_state_dict(state_dict)
        model = model.to(config['device'])
        model.eval()
        return model

    @torch.no_grad()
    def forward(self, x):
        outputs = self.model(x)
        return outputs


def get_data_dict(data):
    data_dict = {}
    for i in range(len(data)):
        data[i] = data[i].strip()
        data[i] = data[i].lower().split(",")
        # merge from data[i][1] to len(data[i])
        data[i][1] = ",".join(data[i][1:])
        data[i][1] = preprocess(data[i][1])
        data_dict[i] = [data[i][1], data[i][0]]
    return data_dict

def elmo(config):
        # read the data
    # get traindata from train.csv
    train_file = open("./data/train.csv", "r")
    test_file = open("./data/test.csv", "r")
    # train_file = open("./data/testing.csv", "r")
    # test_file = open("./data/testing.csv", "r")
    train_data = train_file.readlines()
    test_data = test_file.readlines()
    # skip the first line
    train_data = train_data[1:]
    test_data = test_data[1:]
    # randomly select 5 samples from the train data
    random.seed(config['seed'])
    # keep first num_val_samples as val data
    val_data = train_data[:config['num_val_samples']]
    train_data = list(set(train_data) - set(val_data))
    test_data = test_data[1:]
    train_data_dict = get_data_dict(train_data)
    # randomly select 5 samples from the train data
    val_data_dict = get_data_dict(val_data)
    test_data_dict = get_data_dict(test_data)
    vocab = create_vocab(train_data_dict.values())
    # create the word_to_ix dictionary
    word_to_ix = {}
    for word in vocab:
        word_to_ix[word] = len(word_to_ix)
    ix_to_word = {v:k for k,v in word_to_ix.items()}
    config["ix_to_word"] = ix_to_word
    config["word_to_ix"] = word_to_ix
    config["output_dim"] = len(word_to_ix)
    config["vocab_size"] = len(word_to_ix)

    print(f"Number of words in the vocabulary: {len(vocab)}")
    # create the dataset
    train_dataset = Dataset(train_data_dict,config)
    val_dataset = Dataset(val_data_dict,config)
    test_dataset = Dataset(test_data_dict,config)
    # print(f"Dataset:{train_data_dict}")
    # create the dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    # get the vocab size
    model = LSTM(config)
    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = str(device)
    model.to(config['device'])
    # train the model
    model = train_nn(model, config, train_dataloader,val_dataloader)
    # get the perplexity of the model on the test data
    loss = get_loss(model, config, test_dataloader)
    print(f"Loss on the test set: {loss}")
    embedding_model = ELMO_loaded(config)
    embedding_model.to(config['device'])
    # Just for verification
    loss = get_loss(embedding_model, config, test_dataloader)
    print(f"Loss on the test set: {loss}")
    # save the config dictionary as a json file
    with open(f"./config.json", "w") as f:
        json.dump(config, f)
    # get the embeddings
    # embeddings = get_embeddings(embedding_model, config, test_dataloader)
    

def sweep_agent_manager():
    wandb.init()
    config = dict(wandb.config)
    print(config)
    config['model_name'] = f"LSTM_{config['optimizer']}_lr_{config['lr']}_batch_size_{config['batch_size']}_epochs_{config['epochs']}_nl_{config['num_layers']}_hd_{config['hidden_dim']}_ed_{config['embedding_dim']}_sl_{config['seq_len']}"
    run_name = config['model_name']
    wandb.run.name = run_name
    # main(config)

config = {
    'embedding_dim': 300,
    'hidden_dim': 150,
    'num_layers': 1,
    'optimizer': 'adam',
    'bidirectional': True,
    'batch_first': True,
    'lr': 0.001,
    'batch_size': 64,
    'epochs': 25,
    'device': 'cuda', 
    'seq_len': 45,
    'model_name': 'elmo_pretraining',
    'num_workers': 8,
    'num_classes': 4,
    'seed': 42,
    'num_val_samples': 10000
}
# elmo(config)

if __name__ == "__main__":
    wandb.login()
    # wandb.agent(sweep_id="lakshmipathi-balaji/anlp/30i3r7av", function=sweep_agent_manager, count=100)
    # wandb.init()
    run_name = f"ELMO_{config['optimizer']}_lr_{config['lr']}_batch_size_{config['batch_size']}_epochs_{config['epochs']}_nl_{config['num_layers']}_hd_{config['hidden_dim']}_ed_{config['embedding_dim']}_sl_{config['seq_len']}"
    run = wandb.init(project="anlp_a2", entity="lakshmipathi-balaji", name=run_name, config=config, reinit=True)
    elmo(config)
    wandb.finish()