import json
import codecs
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.classify import maxent

plt.style.use('ggplot')
ENGLISH_TRIAL = 'tutorial/conll16st-en-01-12-16-trial'
ENGLISH_TRAIN = 'conll16st-en-zh-dev-train_LDC2016E50/conll16st-en-01-12-16-train'
PATH_RAW_DOCS = 'conll16st-en-zh-dev-train_LDC2016E50/conll16st-en-01-12-16-train/raw/'

#SET THE USE VARIABLE TO EITHER ENGLISH TRIAL OR THE ENGLISH TRAIN
USE = ENGLISH_TRAIN

#LOADING DATA
# pdtb_file = codecs.open(USE+'/relations.json', encoding='utf8')
# relations = [json.loads(x) for x in pdtb_file]
# en_parse_dict = json.load(codecs.open(USE+'/parses.json', encoding='utf8'))
# file_names = os.listdir(PATH_RAW_DOCS)
# tfidf_all_files = [codecs.open(PATH_RAW_DOCS+f, encoding='latin-1').read() for f in file_names]
# tfidf = TfidfVectorizer(min_df=1, encoding='latin-1', stop_words='english')
# X = tfidf.fit_transform(tfidf_all_files)

# for file_id in en_parse_dict.keys():
	# for file_sentences in en_parse_dict[file_id]['sentences']:
		# print sent
		# break



# train = [
#     ({'a': 1, 'b': 1, 'c': 1}, 'y'),
#     ({'a': 5, 'b': 5, 'c': 5}, 'x'),
#     ({'a': 0.9, 'b': 0.9, 'c': 0.9}, 'y'),
#     ({'a': 5.5, 'b': 5.4, 'c': 5.3}, 'x'),
#     ({'a': 0.8, 'b': 1.2, 'c': 1}, 'y'),
#     ({'a': 5.1, 'b': 4.9, 'c': 5.2}, 'x')
# ]

train = [({'connective': 'unless', 'pos': 'IN', 'position': 12, 'bla': [1,2,3]}, 1),
 ({'connective': 'But', 'pos': 'CC', 'position': 0, 'bla': [1,3]}, 1),
 ({'connective': 'also', 'pos': 'RB', 'position': 3, 'bla': [1,2,3]}, 1),
 ({'connective': 'until', 'pos': 'IN', 'position': 20, 'bla': [1,2,3]}, 1),
 ({'connective': 'as', 'pos': 'IN', 'position': 6, 'bla': [1,2,3]}, 1),
 ({'connective': 'and', 'pos': 'CC', 'position': 14, 'bla': [1,2,3]}, 1),
 ({'connective': 'until', 'pos': 'IN', 'position': 18, 'bla': [1,2,3]}, 1),
 ({'connective': 'until', 'pos': 'IN', 'position': 6, 'bla': [1,2,3]}, 1),
 ({'connective': 'before', 'pos': 'IN', 'position': 13, 'bla': [1,2,3]}, 1),
 ({'connective': 'Moreover', 'pos': 'RB', 'position': 0, 'bla': [1,2,3]}, 1),
 ({'connective': 'Test', 'pos': 'XX', 'position': 90, 'bla': [1,2,3,6,7]}, 0)
        ]

# test = [
#     {'a': 5.2, 'b': 5.1, 'c': 5},
#     {'a': 1, 'b': 0.8, 'c': 1.2},
#     {'a': 5.2, 'b': 5.1, 'c': 5}
#     ]

test =  [
    {'connective': 'until', 'pos': 'IN', 'position': 18},
    {'connective': 'until', 'pos': 'IN', 'position': 6},
    {'connective': 'before', 'pos': 'IN', 'position': 13},
    {'connective': 'Moreover', 'pos': 'RB', 'position': 0},
    {'connective': 'but', 'pos': 'CC', 'position': 16},
    {'connective': 'after', 'pos': 'IN', 'position': 21},
    {'connective': 'Testing', 'pos': 'XXX', 'position': 90}
  ]

encoding = maxent.TypedMaxentFeatureEncoding.train(
    train, count_cutoff=3, alwayson_features=True)

classifier = maxent.MaxentClassifier.train(
    train, bernoulli=False, encoding=encoding, trace=0)

classifier.classify_many(test)