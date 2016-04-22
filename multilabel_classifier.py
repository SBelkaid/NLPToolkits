
import json
import codecs
import os
import pickle
# import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.classify import maxent
from nltk.classify.util import accuracy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
from sklearn.preprocessing import LabelBinarizer
from nltk.classify import MultiClassifierI
from nltk.classify import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer as mlb
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report

# plt.style.use('ggplot')
ENGLISH_TRIAL = 'tutorial/conll16st-en-01-12-16-trial'
ENGLISH_TRAIN = 'conll16st-en-zh-dev-train_LDC2016E50/conll16st-en-01-12-16-train/'
PATH_RAW_DOCS = 'conll16st-en-zh-dev-train_LDC2016E50/conll16st-en-01-12-16-train/raw/'

#SET THE USE VARIABLE TO EITHER ENGLISH TRIAL OR THE ENGLISH TRAIN
USE = ENGLISH_TRAIN
VECTORIZER = CountVectorizer(ngram_range=(1,8), analyzer='word', lowercase=False)
ANALYZER = VECTORIZER.build_analyzer()


def get_part_of_speech(docID, span):
    """
    Get PoS from the parses.json file by filename, word and span.
    """
    sentencelist = enumerate(PARSES[docID]["sentences"])
    for sentenceID, sentence in sentencelist:
        for word_data in sentence["words"]:
            word = word_data[0]                
            off_begin = word_data[1]["CharacterOffsetBegin"]
            off_end = word_data[1]["CharacterOffsetEnd"]
            part_of_speech = word_data[1]["PartOfSpeech"]
            begin, end = span
            if off_begin == begin and off_end == end:
                return part_of_speech

def get_phrase_structure(docID, sentenceID):
    """
    Retrieve phrase_structure from the parses.json file by filename and sentenceID.
    """
    sentencelist = PARSES[docID]["sentences"]
    phrase_structure = sentencelist[sentenceID]["parsetree"]
    return phrase_structure

def PS_or_SS(arg1_sentenceID, sentenceID):
    """
    Returns Arg1 type (PS or SS) based on sentenceIDs.
    """
    if arg1_sentenceID == sentenceID:
        return "SS"
    elif arg1_sentenceID == sentenceID-1:
        return "PS"

def discourseConnectives(return_list=True):
	"""
	return a dictionairy of all the possible connective senses and how they occur
	by putting the words indicating the type in a set.
	"""
	connectives = defaultdict(set)
	list_of_connectives = list()
	for rel in KNOWN_RELATIONS:
		if rel['Type']=='Explicit':
				connectives[rel['Sense'][0]].add(rel['Connective']['RawText'].lower())
	if return_list:
		for key, val in connectives.items():
			for v in val:
				list_of_connectives.append((v, key))
		return list_of_connectives
	return connectives

def constructTrainingData(list_of_relations):
	"""
	returns a list of tuples containing dictioniaries with features.
	"""
	#To find all the connectives and store them in a list
	connectivelist = []
	tuplelist = []
	#iterate through the file:
	for relationID, relation in enumerate(list_of_relations):
		connective = relation["Connective"]["RawText"]
		connective_type = relation["Type"]
		if connective_type != 'Explicit':
			continue
		docID = relation["DocID"]
		tokenlist = relation["Connective"]["TokenList"]
		sense = relation["Sense"][0]
		connective_extent = relation["Connective"]["CharacterSpanList"][0]
		tokenID = tokenlist[0][2]
		sentenceID = tokenlist[0][3]
		sentence_position = tokenlist[0][4]

		arg1 = relation["Arg1"]
		arg1_extent = arg1["CharacterSpanList"][0]
		arg1_sentenceID = arg1["TokenList"][0][3]
		arg1_type = PS_or_SS(arg1_sentenceID, sentenceID) #get arg1_type
		if arg1_type == "PS":
			arg1_phrase_structure = get_phrase_structure(docID, arg1_sentenceID)
		else:
			arg1_phrase_structure = None
		arg2 = relation["Arg2"]
		arg2_extent = arg2["CharacterSpanList"][0]
		arg2_sentenceID = arg2["TokenList"][0][3]
		part_of_speech = get_part_of_speech(docID, connective_extent)
		phrase_structure = get_phrase_structure(docID, sentenceID)
		# print docID, sentenceID, part_of_speech, connective
		dependency_heading, dependency_attached = get_connective_dependency(docID, sentenceID, connective)
		connectivelist.append(connective.lower())
		features = {'connective': connective,
		 # 'pos': part_of_speech,
		 'position': sentence_position,
		 'dependency':dependency_heading,
		 'phrase_structure':phrase_structure}

		label = relation['Sense']
		tuplelist.append((features, label))
	return tuplelist

def get_connective_dependency(docID, sentenceID, connective):
    """
    Return heading and attached dependencies of connectives. 
    """
    dependency_heading = "_"
    dependency_attached = "_"
    sentencelist = PARSES[docID]["sentences"]
    dependencylist = sentencelist[sentenceID]["dependencies"]
    for dependency in dependencylist:
        pos, heading, attached = dependency
        heading_token, headingID = heading.split("-")[:2]
        attached_token, attachedID = attached.split("-")[:2]
        if attached_token == connective:
            dependency_heading = heading
        if heading_token == connective:
            dependency_attached = attached

    return dependency_heading, dependency_attached


if __name__ == '__main__':
	print "Loading training data "
	KNOWN_RELATIONS = [json.loads(line) for line in open(ENGLISH_TRAIN+'relations.json', 'r')]
	print "Loading parsed data"
	PARSES = json.load(open(ENGLISH_TRAIN+'parses.json', 'r'))
	print 'DONE'
	print 'Constructing training data'
	training_data = constructTrainingData(KNOWN_RELATIONS)
	print 'DONE'
	X_train, target = zip(*training_data)
	print 'Vectorizing data'
	vec = DictVectorizer()
	train = vec.fit_transform(X_train)
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, target, test_size=0.2)
	clf = OneVsRestClassifier(LogisticRegression())
	mb = mlb()
	y_train = mb.fit_transform(y_train)
	y_test = mb.transform(y_test)
	print 'Training classifier'
	clf.fit(X_train,y_train)
	print 'DONE'
	print 'Classifying {} test samples'.format(y_test.shape[0])
	predicted = clf.predict(X_test)
	print classification_report(y_test, predicted)
	print "Accuracy: {}".format(accuracy_score(y_test, predicted))
	# print "Recall: {}".format(recall_score(y_test, predicted))





