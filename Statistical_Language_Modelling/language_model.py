# implementing witten-bell smooting after cleaning the corpus given as a argument to the program using regex handiling all cases of punctuations and numbers and special characters and links and all and then applying witten-bell smooting to the cleaned corpus after finding n-grmas and then calculating the probability of each n-gram and then calculating the perplexity of the test data given as a argument to the program. This is done in various functions as mentioned by the name of the corpus
import re
import numpy as np
import argparse

# TOKENIZATION
class tokenization:
    def __init__(self):
        pass

    def replaceHashtags(self, txt):
        return re.sub('\#[a-zA-Z]\w+', '<HASHTAG>', txt)

    def replace_email(self, corpus):
        return re.sub(r'\S*@\S*\s?', r'<EMAIL>', corpus)

    def replaceURL(self, txt):
        return re.sub(r'(https?:\/\/|www\.)?\S+[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}\S+', r'<URL>', txt)

    def replaceMentions(self, txt):
        return re.sub(r'@\w+', r'<MENTION>', txt)

    def replaceDateTime(self, txt):
        txt = re.sub(
            r'\d{2,4}\-\d\d-\d{2,4}|\d{2,4}\/\d\d\/\d{2,4}|\d{2,4}:\d\d:?\d{2,4}', '<DATE>', txt)
        return re.sub(r'\d+:\d\d:?\d{0,2}?( am|am| pm|pm)', r'<TIME>', txt)

    # for better results as we go by string comparisions
    def upperToLower(self, txt): return txt.lower()


    def replacePunctuation(self, txt): return re.sub(
        r'(!|"|\#|\$|%|&|\'|\(|\)|\*|\+|,|-|—|’|\.|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|‘|\{|\||\}|~{1,})', r' <PUN> ', txt)

    def replaceMobileNumber(self, txt):
        return re.sub(r'[\+0-9\-\(\)\.]{3,}[\-\.]?[0-9\-\.]{3,}', r'<MOB>', txt)

    def replaceNumericals(self, txt):
        # returning the numericals as <NUM> as we are not considering the numericals in the corpus
        return re.sub(r'\d+', r'<NUM>', txt)

def tokenize(corpus):
    token = tokenization()
    corpus = token.replaceHashtags(corpus)
    corpus = token.replace_email(corpus)
    corpus = token.replaceURL(corpus)
    corpus = token.replaceMentions(corpus)
    corpus = token.upperToLower(corpus)
    corpus = token.replaceDateTime(corpus)
    corpus = token.replacePunctuation(corpus)
    corpus = token.replaceMobileNumber(corpus)
    corpus = token.replaceNumericals(corpus)
    return corpus.split()


# implementing unigram function
# takes cleaned corpus as argument
# returns unigram dictionary with unigram and its count

def unigram(word_list):
    unigram = {}
    unigram_real = {}
    for sentence in word_list:
        for word in sentence:
            if word not in unigram:
                unigram[word] = 1
                unigram_real[word] = 1
            else:
                unigram[word] += 1
                unigram_real[word] = 1
    MIN_COUNT_UNIGRAM = 2
    # removing if any word is less than 10 to "<unk>"
    unigram_real['<unk>'] = 0
    for word in unigram.keys():
        if unigram[word] < MIN_COUNT_UNIGRAM:
            del unigram_real[word]
            unigram_real['<unk>'] += unigram[word]
    return unigram_real


# implementing bigram function
# takes cleaned corpus as argument
# returns bigram dictionary with bigram and its count
def bigram(word_list):
    bigram = {}
    bigram_real = {}
    for sentence in word_list:
        sentence = ['<unk>'] + sentence
        for i in range(len(sentence)-1):
            if sentence[i] not in bigram.keys():
                bigram[sentence[i]] = {}
                bigram[sentence[i]][sentence[i+1]] = 1
                bigram_real[sentence[i]+' ' + sentence[i+1]] = 1
            else:
                if sentence[i+1] not in bigram[sentence[i]].keys():
                    bigram[sentence[i]][sentence[i+1]] = 1
                    bigram_real[sentence[i]+' ' + sentence[i+1]] = 1
                else:
                    bigram[sentence[i]][sentence[i+1]] += 1
                    bigram_real[sentence[i]+" "+sentence[i+1]] += 1
    return bigram_real

# implementing trigram function
# takes cleaned corpus as argument
# returns trigram dictionary with trigram and its count


def trigram(word_list):
    trigram = {}
    trigram_real = {}
    for sentence in word_list:
        sentence = ['<unk>', '<unk>'] + sentence
        for i in range(len(sentence)-2):
            if sentence[i] not in trigram:
                trigram[sentence[i]] = {}
                trigram[sentence[i]][sentence[i+1]] = {}
                trigram[sentence[i]][sentence[i+1]][sentence[i+2]] = 1
                trigram_real[sentence[i] + " " +
                             sentence[i+1] + " " + sentence[i+2]] = 1
            else:
                if sentence[i+1] not in trigram[sentence[i]]:
                    trigram[sentence[i]][sentence[i+1]] = {}
                    trigram[sentence[i]][sentence[i+1]][sentence[i+2]] = 1
                    trigram_real[sentence[i] + " " +
                                 sentence[i+1] + " " + sentence[i+2]] = 1
                else:
                    if sentence[i+2] not in trigram[sentence[i]][sentence[i+1]]:
                        trigram[sentence[i]][sentence[i+1]][sentence[i+2]] = 1
                        trigram_real[sentence[i] + " " +
                                     sentence[i+1] + " " + sentence[i+2]] = 1
                    else:
                        trigram[sentence[i]][sentence[i+1]][sentence[i+2]] += 1
                        trigram_real[sentence[i] + " " +
                                     sentence[i+1] + " " + sentence[i+2]] += 1
    return trigram_real

# implementing fourgram function
# takes cleaned corpus as argument
# returns fourgram dictionary with fourgram and its count


def fourgram(word_list):
    fourgram = {}
    fourgram_real = {}
    for sentence in word_list:
        sentence = ['<unk>', '<unk>', '<unk>'] + sentence
        for i in range(len(sentence)-3):
            if sentence[i] not in fourgram:
                fourgram[sentence[i]] = {}
                fourgram[sentence[i]][sentence[i+1]] = {}
                fourgram[sentence[i]][sentence[i+1]][sentence[i+2]] = {}
                fourgram[sentence[i]][sentence[i+1]
                                      ][sentence[i+2]][sentence[i+3]] = 1
                fourgram_real[sentence[i] + " " + sentence[i+1] +
                              " " + sentence[i+2] + " " + sentence[i+3]] = 1
            else:
                if sentence[i+1] not in fourgram[sentence[i]]:
                    fourgram[sentence[i]][sentence[i+1]] = {}
                    fourgram[sentence[i]][sentence[i+1]][sentence[i+2]] = {}
                    fourgram[sentence[i]][sentence[i+1]
                                          ][sentence[i+2]][sentence[i+3]] = 1
                    fourgram_real[sentence[i] + " " + sentence[i+1] +
                                  " " + sentence[i+2] + " " + sentence[i+3]] = 1
                else:
                    if sentence[i+2] not in fourgram[sentence[i]][sentence[i+1]]:
                        fourgram[sentence[i]][sentence[i+1]
                                              ][sentence[i+2]] = {}
                        fourgram[sentence[i]][sentence[i+1]
                                              ][sentence[i+2]][sentence[i+3]] = 1
                        fourgram_real[sentence[i] + " " + sentence[i+1] +
                                      " " + sentence[i+2] + " " + sentence[i+3]] = 1
                    else:
                        if sentence[i+3] not in fourgram[sentence[i]][sentence[i+1]][sentence[i+2]]:
                            fourgram[sentence[i]][sentence[i+1]
                                                  ][sentence[i+2]][sentence[i+3]] = 1
                            fourgram_real[sentence[i] + " " + sentence[i+1] +
                                          " " + sentence[i+2] + " " + sentence[i+3]] = 1
                        else:
                            fourgram[sentence[i]][sentence[i+1]
                                                  ][sentence[i+2]][sentence[i+3]] += 1
                            fourgram_real[sentence[i] + " " + sentence[i+1] +
                                          " " + sentence[i+2] + " " + sentence[i+3]] += 1
    return fourgram_real


def count_occurence(all_dict, n, string):
    if(len(string) == 0):
        return 0
    try:
        return all_dict[n][string]
    except KeyError:
        return 0


def sum_occurence_count(all_dict, n, string):
    if(len(string) == 0):
        return 0
    sum = 0
    for ele in all_dict[n]:
        if(ele.startswith(string)):
            sum += all_dict[n][ele]
    return sum

def count_unique_occurence(all_dict, n, string):
    if(len(string) == 0):
        return 0
    sum = 0
    for ele in all_dict[n]:
        if(ele.startswith(string)):
            sum += 1
    return sum

def cont_count(all_dict, n, string):
    cnt = 0
    for ele in all_dict[n].keys():
        if ele.split(" ")[-1] == string:
            cnt += 1
    return cnt


def witten_bell(history, novel, all_dict):

    if novel not in all_dict[1].keys():
        novel = '<unk>'
        
    n = len(history.split())+1

    # base case when history is null
    if n == 1:
        if novel in all_dict[1]:
            return((all_dict[1][novel]+1)/(sum(all_dict[1].values()) + len(all_dict[1])))
        else:
            return(1/(sum(all_dict[1].values()) + len(all_dict[1])))

    try:
        lamba = count_unique_occurence(all_dict,n,history)/(count_unique_occurence(all_dict,n,history) + sum_occurence_count(all_dict, n, history))
    # if incase denominator is zero
    except ZeroDivisionError:
        lamba = 1/len(all_dict[n])
    # print(lamba)
    # Maximal Likelihood probability = (count of history + novel)/(count of history)
    try:
        pML = count_occurence(all_dict, n, history+" "+novel)/(sum_occurence_count(all_dict, n, history))
    except ZeroDivisionError:
        pML = 0
    return (1-lamba)*pML + lamba*witten_bell(" ".join(history.split()[1:]), novel, all_dict)


# DOUBT - if the history is unseen in or the sequence is unseen does it make the probabilith 0 acc to this algo
# This DOUBT is from Formula
def kneser_ney(history, novel, recur_step, all_dict):

    if novel not in all_dict[1].keys():
        novel = '<unk>'

    n = len(history.split())+1
    # Considering the discount value as 0.75
    d = 0.75
    # base case when history is null

    # ------------------ Base case when history is null ------------------
    if n == 1:
        return(0.25/len(all_dict[1])) + (0.75/(all_dict[1]['<unk>']))
    # c_KN = count or continuation_count
    if recur_step == 1:
        # Initial case - Highest Order n-gram
        try:
            first_term = max(count_occurence(all_dict, n, " ".join(
                [history, novel]))-0.75, 0)/sum_occurence_count(all_dict, n, history)
        except ZeroDivisionError:
            # Handle 0/0 form
            first_term = 0
    else:
        try:
            first_term = max(cont_count(all_dict, n, novel) - 0.75, 0)/len(all_dict[n-1])
        except ZeroDivisionError:
            first_term = 0

    # -------------Defining Lambda----------------
    try:
        lamba = (((0.75)/sum_occurence_count(all_dict, n, history))*count_occurence(all_dict, n-1, history)) 
    except ZeroDivisionError:
        # Handle 0/0 form
        # if Lambda is then the whole second term is 0 even after all recunrsions
        # So returning first term
        lamba = 1/len(all_dict[1])
    if(lamba == 0):
        lamba = 0.75/len(all_dict[1])
    sec_term = lamba * kneser_ney(" ".join(history.split()[1:]),
                   novel, recur_step+1, all_dict)
    return first_term + sec_term


def perplexityScore(prob_list):
    """
    Input:
        list containing n-gram probability scores
    """
    return np.power(1/np.prod(prob_list), 1/len(prob_list))


def sent2PPScore(sent, n, smoothing, all_dict):
    # Construct n-grams...
    ngrams_lst = []
    tok = sent.split()
    if len(tok) > 0:
        tokens = ["<unk>" for _ in range(n-1)]+tok
        for k in range(n, len(tokens)):
            # n-gram sentence
            ngrams_lst.append(" ".join(tokens[k-n:k]))
    scores = []
    if smoothing == "k":
        for ng in ngrams_lst:
            lst = ng.split(" ")
            hist, current = " ".join(lst[:-1]), lst[-1]
            scores.append(kneser_ney(hist, current, 1, all_dict))
    elif smoothing == "w":
        for ng in ngrams_lst:
            lst = ng.split(" ")
            hist, current = " ".join(lst[:-1]), lst[-1]
            scores.append(witten_bell(hist, current, all_dict))
            
    else:
        raise ValueError("Please enter a correct smoothing format.")
    # calculate PP scores.
    return perplexityScore(scores)

def sent2Prob(sent, n, smoothing, all_dict):
    # Construct n-grams...
    ngrams_lst = []
    tok = sent.split()
    if len(tok) > 0:
        tokens = ["<unk>" for _ in range(n-1)]+tok
        for k in range(n, len(tokens)):
            # n-gram sentence
            ngrams_lst.append(" ".join(tokens[k-n:k]))
    scores = []
    if smoothing == "k":
        for ng in ngrams_lst:
            lst = ng.split(" ")
            hist, current = " ".join(lst[:-1]), lst[-1]
            scores.append(kneser_ney(hist, current, 1, all_dict))
    elif smoothing == "w":
        for ng in ngrams_lst:
            lst = ng.split(" ")
            hist, current = " ".join(lst[:-1]), lst[-1]
            scores.append(witten_bell(hist, current, all_dict))
    else:
        raise ValueError("Please enter a correct smoothing format.")
    # calculate PP scores.
    return (scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('smoothing', type=str, default='k')
    parser.add_argument('corpus', type=str, default='corpus/1.txt')
    n = 4
    smoothing = parser.parse_args().smoothing
    corpus = parser.parse_args().corpus
    LMindex = 0
    if smoothing == "k":
        LMindex = 1
    elif smoothing == "w":
        LMindex = 2
    LMindex = 4
    # f1 = open(corpus, 'r')
    f1 = open(corpus, 'r')
    corpus1 = f1.read()
    word_list1 = []

    # getting sentences based on "." and "?" and "!" and splitting them into words
    newlines = [".", "?", "!"]
    corpus1 = corpus1.replace("\n", " ")

    for ele in newlines:
        corpus1 = corpus1.replace(ele, ele+"\n")

    for (sentence) in corpus1.splitlines():
        cleaned_list = tokenize(sentence)
        if(len(cleaned_list) != 0):
            word_list1.append(cleaned_list)
    
    ############## word_list1 --> list of list of cleaned words in each sentence ##############

    unigram1 = unigram(word_list1)
    
    bigram1 = bigram(word_list1)

    trigram1 = trigram(word_list1)

    fourgram1 = fourgram(word_list1)

    all_dict = {1: unigram1, 2: bigram1, 3: trigram1, 4: fourgram1}
    # print(len(word_list1))

    #makig list of sentences by merging the words in each sentence in word_list1
    list_of_sentences = []
    for i in range(len(word_list1)):
        word_list1[i] = " ".join(word_list1[i])
        list_of_sentences.append(word_list1[i])

    # splitting the list of sentences into train and test set
    for_test = int(0.2*len(list_of_sentences))
    np.random.seed(11)
    idx = np.random.choice(len(list_of_sentences), for_test, replace=False)
    train, test = [], []
    for id in range(len(list_of_sentences)):
        if id in idx:
            test.append(list_of_sentences[id])
        else:
            train.append(list_of_sentences[id])
    # input for type of smoothing
    pp_score_lst_train = []
    pp_score_lst_test = []
    tags = ["<unk>","<PUN>","<HASHTAG>","<EMAIL>","<MENTION>","<URL>","<DATE>", "MOB","<NUM>", "<TIME>"]
    # ------------------ Training ------------------
    # print("="*20+"COMPUTATION STARTED FOR TRAINING SET"+"="*20)
    with open("scores/2021114007_LM"+str(LMindex)+"_train-perplexity.txt", "w") as file:
        for j, sent in enumerate(train):
            # Omit the newline character
            sent = sent.strip()
            # If by any chance sentence is of length 0
            if len(sent) != 0 and sent not in tags:
                try:
                    pp_score_lst_train.append(sent2PPScore(sent, n, smoothing, all_dict=all_dict))
                except:
                    print(sent)
                file.write(sent+"    PP Score = {0:.3f}\n".format(pp_score_lst_train[-1]))
                # print(f"Sentence {j+1} is PROCESSED!!!")
    with open("scores/2021114007_LM"+str(LMindex)+"_train-perplexity.txt", "r+") as file:
        prev_content = file.read()
        file.seek(0, 0)
        file.write("Average_Perplexity Score: {0:.3f}\n".format(np.mean(pp_score_lst_train))+ prev_content)

    print("="*50+"DONE!!!"+"="*50)
    

    # ------------------ Testing ------------------
    # print("="*20+"COMPUTATION STARTED FOR TESTING SET"+"="*20)
    with open("scores/2021114007_LM"+str(LMindex)+"_test-perplexity.txt", "w") as file:
        for j, sent in enumerate(test):
            # Omit the newline character
            sent = sent.strip()
            # If by any chance sentence is of length 0
            if len(sent) != 0 and sent not in tags:
                pp_score_lst_test.append(sent2PPScore(sent, n, smoothing, all_dict=all_dict))
                file.write(sent+"    PP Score = {0:.3f}\n".format(pp_score_lst_test[-1]))
                # print(f"Sentence {j+1} is PROCESSED!!!")
    with open("scores/2021114007_LM"+str(LMindex)+"_test-perplexity.txt", "r+") as file:
        prev_content = file.read()
        file.seek(0, 0)
        file.write("Average_Perplexity Score: {0:.3f}\n".format(np.mean(pp_score_lst_test))+ prev_content)

    # print("="*50+"DONE!!!"+"="*50)

    #---------------------Testing on Command Line---------------------
    # print("="*20+"COMPUTATION STARTED FOR TESTING ON COMMAND LINE"+"="*20)
    while True:
        sent = input("input sentence: ")
        if sent == "exit":
            break
        else:
            scores = sent2Prob(sent, n, smoothing, all_dict=all_dict)
            prob = 1
            for i in scores:
                prob = prob * i
            print(prob)