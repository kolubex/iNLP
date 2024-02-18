from .imports import *
def create_vocab(data):
    """
    Args:
        data: the data to create the vocabulary
    Returns:
        vocab: the vocabulary
    """
    vocab = ["unk", "sos", "eos", "pad"]
    for sentence in data:
        sentence = sentence[0]
        sentence = sentence.lower()
        for word in sentence.split():
            if word not in vocab:
                vocab.append(word)
    
    # return the unique words in the vocabulary
    return list(set(vocab))

def preprocess(corpus):
    """
    Args:
        corpus: the corpus to preprocess (str)
    Returns:
        sentences: list of sentences
    """
    cleaned_sentences = []
    sentence_pattern = r'[.?!]'
    
    # Replace or remove special text types:
    corpus = re.sub(r'https?://\S+|www\.\S+', 'URL', corpus)  # Replace URLs with <URL>
    corpus = re.sub(r'\S+@\S+', 'EMAIL', corpus)              # Replace emails with <EMAIL>
    corpus = re.sub(r'\d+(\.\d+)?', 'NUM', corpus)            # Replace numbers with <NUM>
    
    # Use the re.split() function to split the text into sentences
    sentences = re.split(sentence_pattern, corpus)

    # Remove empty strings and leading/trailing whitespace from the sentences
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    for sentence in sentences:
        # make lowercase
        sentence = sentence.lower()
        # Replace punctuations with space
        text = re.sub(r'[^\w\s<>]', r' ', sentence)
        # tokenize the text
        tokens = nltk.word_tokenize(text)
        if(len(tokens)):
            # create a sentence with space separated tokens
            sentence = ' '.join(tokens)
            # add sos and eos tokens to the sentence
            sentence = 'sos ' + sentence + ' eos'
            # append the sentence to the cleaned_sentences list
            cleaned_sentences.append(sentence)
    # merge the cleaned_sentences into a single string
    cleaned_sentences = ' '.join(cleaned_sentences)
    return cleaned_sentences
