# -*- coding: utf-8 -*-

import os.path
import pickle
from itertools import chain

from nltk import NaiveBayesClassifier
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def get_cache_variable(key):
    key = './cache/' + key + '.pickle'
    out = False
    if os.path.exists(key):
        with open(key, 'rb', buffering=-1) as f:
            out = pickle.load(f)
    return out


def set_cache_variable(var, key):
    key = './cache/' + key + '.pickle'
    with open(key, 'wb') as f:
        pickle.dump(var, f)


training_data = [('I love this sandwich.', 'pos'),
                 ('This is an amazing place!', 'pos'),
                 ('I feel very good about these beers.', 'pos'),
                 ('I know Bob.', 'pos'),
                 ('This is my best work.', 'pos'),
                 ('This is my best works.', 'pos'),
                 ("What an awesome view", 'pos'),
                 ('I do not like this restaurant', 'neg'),
                 ('I am tired of this stuff.', 'neg'),
                 ("I can't deal with this", 'neg'),
                 ('He is my sworn enemy!', 'neg'),
                 ('I do not know Bob.', 'neg'),
                 ('My boss is horrible.', 'neg')]

vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in training_data]))
print vocabulary

wnl = WordNetLemmatizer()
word_net = set(wnl.lemmatize(i) for i in vocabulary)
print word_net

feature_set = [
    ({i: (i in word_tokenize(sentence.lower())) for i in vocabulary}, tag) for sentence, tag in training_data
]

# if os.path.exists(cache):
#     with open(cache, 'rb', buffering=-1) as f:
#         classifier = pickle.load(f)
# else:
#     # список сорв в разных формах и признаком присутсвия в общей выборке слов , категория
#     classifier = NaiveBayesClassifier.train(feature_set)
#     # import pickle
#     with open(cache, 'wb') as f:
#         pickle.dump(classifier, f)

classifier = []
classifier = get_cache_variable('classifier')
if classifier is False:
    classifier = NaiveBayesClassifier.train(feature_set)
    set_cache_variable(classifier, 'classifier')

sentences = [
    "This is the best band I've ever heard!",
    "I do not know english!",
    "I know english."
]

for test_sentence in sentences:
    featurized_test_sentence = {i: (i in word_tokenize(test_sentence.lower())) for i in vocabulary}

    print "test_sent:  ", test_sentence
    print "tag:  ", classifier.classify(featurized_test_sentence)
