## Process followed in code implementation

1. Loaded Datasets in the form => (0, 'she', 'PRON')
2. Then Used get_words_and_tags function to get the words and tags in a sequence sentence wise.
3. Used the training data to give indices for the valid and test tags.
4. trained, evaluated and tested the model and tuned hyperparameters to get the best model.
5. Dataloader function or Custom_Dataset function is not used as I made batches of batch_size no.of senteces using slicing
6. Accuracy, F1 score, Confusion Matrix, Classification Report are used to evaluate the model.



## Running the file
You can run the file in 2 ways one by loading the model or by running the file in the terminal.
Please make sure that the datasets are there in the UD_English-Atis folder.

## Test with your sentence
You can test the model with your sentence by running the file in the terminal and then entering the sentence.

>Details of the best model and its parameters are in the report.