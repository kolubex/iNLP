# open a csv file and seperate the data in train.txt into 2 parts and keep in different coloumns
import re
import csv
import pandas as pd
from nltk.tokenize import word_tokenize
# from second coloumn of the csv file get the number of unique words
with open('../datasets/mt_enghinglish/train.csv') as f:
    reader = csv.reader(f)
    # make a dictionary of the words and their frequency
    word_freq = {}
    a = []
    for row in reader:
        row = row[1].split()
        if(row):
            first_word = row[0]
            if first_word not in word_freq:
                word_freq[first_word] = 1
            else:
                word_freq[first_word] += 1

    # sort the dictionary in descending order of frequency
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    # use pandas and print the top 100 words
    df = pd.DataFrame(sorted_word_freq)
    print(sorted_word_freq[:7])
    # make a list of first 7 words in the sorted list
    top_words = [x[0] for x in sorted_word_freq[:7]]
    print(top_words)
    print(sum([x[1] for x in sorted_word_freq[:7]]))
    
# now consider only those rows whose second coloumn does not start with any of the top 14 words
with open('../datasets/mt_enghinglish/train.csv') as f:
    reader = csv.reader(f)
    # make a dictionary of the words and their frequency
    word_freq = {}
    a = []
    for row in reader:
        row1 = row[1].split()
        if(row1):
            first_word = row1[0]
            if first_word not in top_words:
                a.append(row)
    print(len(a))

# now add this to a new csv file
with open('../datasets/mt_enghinglish/train_filtered.csv', 'w') as f:
    # first coloumn input_sentence and second coloumn output_sentence
    writer = csv.writer(f)
    writer.writerows(a)