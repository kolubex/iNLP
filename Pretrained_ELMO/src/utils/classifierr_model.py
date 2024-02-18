from .imports import *
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

class LSTMClassifier(nn.Module):
    def __init__(self, config):
        super(LSTMClassifier, self).__init__()
        self.config = config
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.n_layers = config['num_layers']
        self.bidirectional = config['bidirectional']
        self.batch_first = config['batch_first']
        self.hardcoded = config['hardcoded']
        self.a1 = config['a1']
        self.a2 = config['a2']
        self.a3 = config['a3']
        
        # Initialize raw learnable parameters
        self.raw_a1_learnable = nn.Parameter(torch.randn(1))
        self.raw_a2_learnable = nn.Parameter(torch.randn(1))
        self.raw_a3_learnable = nn.Parameter(torch.randn(1))
        
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.n_layers, batch_first=self.batch_first, bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.hidden_dim*2, self.output_dim)
        self.softmax = nn.Softmax(dim=1)
    
    @property
    def a1_learnable(self):
        return self.raw_a1_learnable / (self.raw_a1_learnable + self.raw_a2_learnable + self.raw_a3_learnable)

    @property
    def a2_learnable(self):
        return self.raw_a2_learnable / (self.raw_a1_learnable + self.raw_a2_learnable + self.raw_a3_learnable)

    @property
    def a3_learnable(self):
        return self.raw_a3_learnable / (self.raw_a1_learnable + self.raw_a2_learnable + self.raw_a3_learnable)

    def forward(self, x):
        if self.hardcoded:
            elmo_embedding = self.a1*x[0] + self.a2*x[1] + self.a3*x[2]
        else:
            elmo_embedding = self.a1_learnable*x[0] + self.a2_learnable*x[1] + self.a3_learnable*x[2]
        
        lstm_out, (hidden_states, cell_states) = self.lstm(elmo_embedding)
        cell_states = cell_states[-2:,:,:].permute(1,0,2)
        cell_states = torch.cat((cell_states[:,0,:], cell_states[:,1,:]), dim=1)
        logits = self.fc(cell_states)
        normalized_logits = self.softmax(logits)
        return {
            "logits": logits,
            "normalised_logits": normalized_logits,
            "elmo_embeddings": elmo_embedding
        }


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
