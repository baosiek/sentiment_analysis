import pandas as pd
import numpy as np
import matplotlib as mpl
import time
from sentiment_network import SentimentNetwork

reviews_file_path = 'data/reviews.txt' # path to reviews
labels_file_path = 'data/labels.txt' # path to labels

# Parameters
cut_off = 10 # no. of revies to print
word_freq_threshold = 100 # minimum word count to consider it a common word

def pretty_reviews_labels_printing(pol, review):
    print(f'{pol}\t\t{review[:120]}...')

def println(string):
    print(str(string) + '\n')

# Reviews dataset
reviews_file = open(reviews_file_path, 'r') # The input
reviews = list(map(lambda x: x.rstrip(), reviews_file.readlines()))
reviews_file.close()
print(f'Number of reviews: [{len(reviews)}]')

# labels dataset
labels_file = open(labels_file_path, 'r') # The outut
labels = list(map(lambda x: x.rstrip(), labels_file.readlines()))
labels_file.close()
print(f'Number of labels: [{len(reviews)}]')

# "Developing a theory"
# There is a pattern in the word distribution of positive reviews
# that is different from the pattern of negative reviews

# Starting random number generator
np.random.seed(123)
count = 0;
# 5 positive reviews
while count < cut_off:
    i = np.random.randint(0, len(labels), size=1)
    if labels[i[0]] == 'positive':
        count += 1
        pretty_reviews_labels_printing(labels[i[0]], reviews[i[0]])

print('\n')

count = 0
while count < cut_off:
    i = np.random.randint(0, len(labels), size=1)
    if labels[i[0]] == 'negative':
        pretty_reviews_labels_printing(labels[i[0]], reviews[i[0]])
        count += 1

# Project 1: Quick Theory Validation
from collections import Counter

# To prove the theory we will compute the positive to negative ratio
positive_vocab_freq = Counter()
negative_vocab_freq = Counter()
total_vocab_freq = Counter()

for index in range(len(labels)):
    if labels[index] == 'positive':
        tokens = reviews[index].split()
        for token in tokens:
            positive_vocab_freq[token] += 1
            total_vocab_freq[token] += 1

for index in range(len(labels)):
    if labels[index] == 'negative':
        tokens = reviews[index].split()
        for token in tokens:
            negative_vocab_freq[token] += 1
            total_vocab_freq[token] += 1

print('/n')

# Word ratio of most common words
pos_neg_ratio = Counter()
for token, freq in list(total_vocab_freq.most_common()):
    if freq > word_freq_threshold:
        pos_neg_ratio[token] = np.log(positive_vocab_freq[token] / \
                               (negative_vocab_freq[token] + 1)) # +1 for denominator != 0

# Words that appears more in positive context than negative have ratio much bigger
# than one. Words that appears more in negative context have ratio between 0 and 1
# much closer to zero. Words that appears equally in both contexts have ratio close to one
print(f'Ratio for amazing: [{pos_neg_ratio["amazing"]}]')
print(f'Ratio for terrible: [{pos_neg_ratio["terrible"]}]')
print(f'Ratio for the: [{pos_neg_ratio["the"]}]')

# Project 2: Creating the Input/Output Data

# The vocabulary
vocab = set(total_vocab_freq.keys());
println(f'Vocabulary size is [{len(vocab)}]. Vocab type is [{type(vocab)}]')

# Initializing layer_0
layer_0 = np.zeros((1, len(vocab)))
println(f'layer_0 shape {layer_0.shape}')

# Token 2 index
token_2_index = {}
for index, token in enumerate(vocab):
    token_2_index[token] = index;


def update_input_layer(review):
    """ Modify the global layer_0 to represent the vector form of review.
    The element at a given index of layer_0 should represent
    how many times the given word occurs in the review.
    Args:
        review(string) - the string of the review
    Returns:
        None
    """

    global layer_0

    # clear out previous state, reset the layer to be all 0s
    layer_0 *= 0

    # count how many times each word is used in the given review and store the results in layer_0
    for token in review.split():
        layer_0[0][token_2_index[token]] += 1

update_input_layer(reviews[0])

def get_target_for_label(label):
    """Convert a label to `0` or `1`.
    Args:
        label(string) - Either "POSITIVE" or "NEGATIVE".
    Returns:
        `0` or `1`.
    """
    if(label == 'positive'):
        return 1
    else:
        return 0

print(f'Target for review [0] is \'{labels[0]}\' -> [{get_target_for_label(labels[0])}]')
print(f'Target for review [1] is \'{labels[1]}\' -> [{get_target_for_label(labels[1])}]')
print(f'Target for review [2] is \'{labels[2]}\' -> [{get_target_for_label(labels[2])}]')

# Project 3: Building a Neural Network
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.001)
mlp.train(reviews[:-1000],labels[:-1000])





