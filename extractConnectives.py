import json
import codecs
import os
import pandas as pd
# import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.classify import maxent
from collections import deque

# plt.style.use('ggplot')
ENGLISH_TRIAL = 'tutorial/conll16st-en-01-12-16-trial'
ENGLISH_TRAIN = 'conll16st-en-zh-dev-train_LDC2016E50/conll16st-en-01-12-16-train/'
PATH_RAW_DOCS = 'conll16st-en-zh-dev-train_LDC2016E50/conll16st-en-01-12-16-train/raw/'

#SET THE USE VARIABLE TO EITHER ENGLISH TRIAL OR THE ENGLISH TRAIN
USE = ENGLISH_TRAIN

def readFile(path_to_json):
	with open(path_to_json) as input_file:
		all_input = json.load(input_file)
	return all_input

def get_part_of_speech(docID, find_word, span):
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

def constructTrainingData(list_of_relations):
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
		part_of_speech = get_part_of_speech(docID, connective, connective_extent)
		phrase_structure = get_phrase_structure(docID, sentenceID)
		# print docID, sentenceID, part_of_speech, connective
		dependency_heading, dependency_attached = get_connective_dependency(docID, sentenceID, part_of_speech, connective)
		connectivelist.append(connective.lower())
		# print(docID, #filename
		# 	tokenID, #unique token ID
		# 	sentenceID, #sentenceID
		# 	sentence_position, #position in sentence
		# 	connective, #token
		# 	connective_extent, #extent connective
		# 	part_of_speech, #PoS
		# 	dependency_heading, #what is the head of the connective
		# 	dependency_attached, #what is attached to the connectiv
		# 	phrase_structure, #phrase structure
		# 	arg1_extent, #extent arg1
		# 	arg1_type, #PS or SS
		# 	arg1_phrase_structure, #arg1 phrase structure
		# 	arg2_extent, #extent arg2
		# 	sense, #meaning
		# 	relationID #discourse relation ID
		# 	)
		tuplelist.append(({'connective': connective, 'pos': part_of_speech, 'position': sentence_position}, 1))
	return tuplelist

def get_connective_dependency(docID, sentenceID, part_of_speech, connective):
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
	KNOWN_RELATIONS = [json.loads(line) for line in open(ENGLISH_TRAIN+'relations.json', 'r')]
	PARSES = readFile(ENGLISH_TRAIN+'parses.json')
	training_data = constructTrainingData(KNOWN_RELATIONS)

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
	    training_data, count_cutoff=3, alwayson_features=True)

	classifier = maxent.MaxentClassifier.train(
	    training_data, bernoulli=False, encoding=encoding, trace=0)

	print classifier.classify_many(test)