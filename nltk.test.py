from itertools import chain

from nltk import NaiveBayesClassifier as nbc
from nltk.tokenize import word_tokenize

training_data = [('I love this sandwich.', 'pos'),
                 ('This is an amazing place!', 'pos'),
                 ('I feel very good about these beers.', 'pos'),
                 ('I know Bob.', 'pos'),
                 ('This is my best work.', 'pos'),
                 ("What an awesome view", 'pos'),
                 ('I do not like this restaurant', 'neg'),
                 ('I am tired of this stuff.', 'neg'),
                 ("I can't deal with this", 'neg'),
                 ('He is my sworn enemy!', 'neg'),
                 ('I do not know Bob.', 'neg'),
                 ('My boss is horrible.', 'neg')]

vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in training_data]))

feature_set = [
    ({i: (i in word_tokenize(sentence.lower())) for i in vocabulary}, tag) for sentence, tag in training_data
]

classifier = nbc.train(feature_set)

sentences = [
    "This is the best band I've ever heard!",
    "I do not know english!",
    "I know english."
]

for test_sentence in sentences:

    featurized_test_sentence = {i: (i in word_tokenize(test_sentence.lower())) for i in vocabulary}

    print "test_sent:", test_sentence
    print "tag:", classifier.classify(featurized_test_sentence)
