## ASSIGNMENT 4
### Pretraining model
- Class Elmo extended from nn.Module is the implementation of elmo architecture with referring the paper - https://arxiv.org/abs/1802.05365

- main() contains the code for pretraining and one can choose the files to run pretraining which makes the code extendable to other datasets as well.

- Once you get the hiddens states of every word in a given sentence from the pretraining model, you can use it for any downstream task. For example, in the code, I have used it for sentiment analysis task.

### Downstream task
- Class SST extended from nn.Module.
- In SST class we get word embeddings from the pretraining model and sum it as a form of linear equation where the co-efficients are learnable parameters.
- And concatenate it with the wordtovec embeddings of the word and pass it through a lstm layer.
- From the output of lstm layer we get the hidden states of every word in a sentence and we pass it through a linear layer to get a tensor of no. of classes, we log_softmax and train it.
- sst() function trains the model and saves the model with best validation accuracy and tests it on the test set where we get classification report.

### Instructions to run the code
- To run the code, you need to have the following files in the sst sub-directory of the code:
    -train.csv
    -test.csv
    -val.csv

- Run python3 code.py, refer the colab notebook for more details.

    
