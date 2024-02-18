# Intorduction
- The code is implementation ofthe paper `Deep contextualized word representations` by Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee and Luke Zettlemoyer. 
- The paper introduces a new type of word representation that captures both complex characteristics of word use (e.g., syntax and semantics), and how these uses vary across linguistic contexts (i.e., to model polysemy). 
- The model is called ELMo (Embeddings from Language Models), which learns from the internal states of a deep bidirectional language model (biLM), which is pre-trained on a large text corpus. 
- The pre-trained stacked BiLSTM is then used to produce contextualized word embeddings as input to downstream NLP tasks.
- The code is implemented in PyTorch and uses the `torch`, `nltk`.. etc libraries.
- The codebase also uses the `wandb` library for logging the results and hyperparameters.
- The code is implemented in a modular way and is easy to understand. The code is written in a way that it can be easily extended to other models and datasets.

> Please drop a ⭐ if you like the work or found it useful and feel free to fork the repository.
The file structure is as follows:
```
.
├── README.md
├── Report.pdf
└── src
    ├── classifier_nnparam.py
    ├── classifier.py
    ├── elmo.py
    └── utils
        ├── classifierr_model.py
        ├── data_classifier.py
        ├── data_elmo.py
        ├── imports.py
        └── preprocessing.py

2 directories, 10 files
```

## 2. How to run the code
* Install the requirements in a new environment.
```
pip install -r requirements.txt
```
* Change directoy to src directory.
```
cd src
```
* Run the code.
```bash
python elmo.py  # For ELMO pretraining
python classifier.py  # For downstream task
# Note that we model has to be downloaded and stored in the correct folder to run the code, and data has to be respective folder (data) in the same directory as the code.
# And please initalise with a sweep in wandb if you want to run the code without wandb then run only main function in the code.

```
* The models need for the embeddings will be autodownloaded when you run the code.
* Please mention the path that you want the model to be save in the code.
* You can find the models and embeddings that are saved here: [Models](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/lakshmipathi_balaji_research_iiit_ac_in/EvtTVQTb2DVIsoGL6_g9yroB4oaKe6OIiVe6VVP8AkiUkw?e=TiSwZZ)
* If you want to load them and run just use the following code.
```py
import torch
model = torch.load('path_to_model')
# given it is inside the script given in the src folder as it needs the class definition.
```
* Link to WandB results: [WandB](https://wandb.ai/lakshmipathi-balaji/anlp_a2?workspace=user-lakshmipathi-balaji)

