# Introduction
This is a collection of the coursework of `intro to NLP course in IIITH taught by Prof. Manish Shrivastava.`

> Please drop a ⭐ if you like the work or found it useful and feel free to fork the repository. And feel free to raise an issue if you find any bugs in the repo.
## 1. List of Assignments
1. Statistical Language Modelling using Kneseer Ney Smoothing and Witten Bell Smoothing. It also includes the implementation of LSTM based language model.
2. POS tagging using LSTM.
3. Word2Vec using CBOW with and without negative sampling and SVD.
4. Pretrained ELMO model for downstream task.

## 2. Course Project
1. The course project is on Code Mixed Language Generation. The code and report are included in the folder.

## 3. Usage
1. Every folder has its own readme file which explains the process followed in the code implementation and how to run the code.

2. To clone the repo - 
```bash
git clone 
```
3. To run the code, follow the instructions in the respective readme files.
4. The subfolders also have datasets and models used in the code. The models are stored in the respective folders and the datasets are stored in the respective folders.
5. The code is also integrated with wandb for logging the results and hyperparameters. The link to the wandb results is also provided in the respective readme files.
6. The code is written in a modular way and is easy to understand. The code is written in a way that it can be easily extended to other models and datasets.


Structure of the Repository.
```
.
├── CodeMixed_generation
│   ├── code
│   ├── Presentation.pdf
│   ├── README.md
│   └── Report.pdf
├── POS_tagging_LSTM
│   ├── Assignment-2.pdf
│   ├── model_300_256_4_2_3.pt
│   ├── pos_tagger.py
│   ├── readme.md
│   ├── Report.pdf
│   └── UD_English-Atis
├── Pretrained_ELMO
│   ├── Elmos_assignment.pdf
│   ├── README.md
│   ├── Report.pdf
│   └── src
├── README.md
├── Statistical_Language_Modelling
│   ├── corpus
│   ├── language_model.py
│   ├── LSTM.py
│   ├── model1.pth
│   ├── model.pth
│   ├── README.md
│   ├── report.pdf
│   └── scores
└── Word_to_Vec_CBOW_SVD
    ├── cbow_ns.py
    ├── cbow.py
    ├── readme.md
    ├── report.pdf
    ├── svd.py
    └── test.json
```
