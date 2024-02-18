# Smoothing-and-LSTM
This folder is implementaion of Witten-bell smoothing, Kenser-ney smoothing, LSTM implementation using pytorch. This is a part of the coursework of intro to NLP course in IIITH taught by Prof. Manish Shrivastava.

> Please drop a ⭐ if you like the work or found it useful and feel free to fork the repository.

## Requirements

- Python 3.6
- Pytorch 0.4.0
- Numpy

## Usage

### Training

```bash
$ python3 langugae_model.py k /corpus/1.txt
```
This command will train a language model using kenser-ney smoothing and the perplexity in scores/2021114007_LM1_train-perplexity.txtand 
2021114007_LM1_test-perplexity.txt

The model files are stored in .pth format in current directory as `./model.pth` and `./model1.pth`.

Find the perplexity score of the model in the scores folder, and find other details in `./report.pdf`.