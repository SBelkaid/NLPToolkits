"""
This script builds a binairy classifier to classify cue phrases. Candidate cue phrases are generated using n-grams. 
The candidates are compared to real cue phrases from the training data if they don't occure in that list then they're used 
as negative examples to train the classifier. 
"""

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

		tuplelist.append((features, 1))
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

def mapCandidates(tokens, candidates):
	"""
	iterates through the tokens and offset parameter and finds the offset of the possible candidate.
	This has been done because after making the ngrams the original token data is lost and requires
	mapping. First the candidate cues are sorted based on length of phrase, this is done because 
	an 1-gram can refer to the same word in an ngram so they can't be seen as seperate. Therefore,
	the longer cues are extracted first. 
	"""
	identified_candidates = []
	duplicate_real_cues = []
	for candi in sorted(candidates, key=lambda x: len(x.split()), reverse=True):
		candi_tokens = candi.split()
		if len(candi_tokens) > 1:
			phrase = []
			for word in candi_tokens:
				if word in zip(*tokens)[0]:
					# print tokens[zip(*tokens)[0].index(word)]
					phrase.append(tokens[zip(*tokens)[0].index(word)])
					removed = tokens.pop(zip(*tokens)[0].index(word))
					# duplicate_real_cues.append(removed)

			if phrase:
				offset = [[phrase[0][1][0], phrase[-1][1][-1]]]
				restructured = ' '.join(zip(*phrase)[0]), offset
				identified_candidates.append(restructured)

		else:
			for token in zip(*tokens)[0]:
				if candi == token:
					# print candi, tokens[zip(*tokens)[0].index(candi)][1]
					identified_candidate = (candi, [tokens[zip(*tokens)[0].index(candi)][1]])
					identified_candidates.append(identified_candidate)
					tokens.remove(tokens[zip(*tokens)[0].index(candi)])
	return identified_candidates, duplicate_real_cues

def comparePhrases(sorted_ngrams, mapping_connectives):
	"""
	Compares the candidates generated by extractCandidates to the real cue phrases in KNOWN_RELATIONS.
	If the cue is in the KNOWN_RELATIONS the it is removed from the candidate list. What is left over is
	used as the negative samples for the training data. 
	"""
	candidates = []
	for gram in sorted_ngrams:
		if gram in zip(*mapping_connectives)[0]:
			candidates.append(gram)
	return candidates

def returnRealConnectives(doc_ID):
	"""
	print all the actual connectives in a document.
	"""
	real_connectives = []
	for rel in KNOWN_RELATIONS:
		if rel['DocID']== doc_ID and rel['Type'] == 'Explicit':
			real_connectives.append((rel['Connective']['RawText'],rel['Connective']['CharacterSpanList']))
			# print rel['Connective']['RawText'], rel['Connective']['CharacterSpanList']
	return real_connectives

def returnNegativeSamples(real_cue_list, candidate_cue_list):
	"""
	compare the candidate cues with the real cues, if it's the same then it will be removed rom the candidate list.
	It will return a list of negative samples of cue phrases. In order for the binary classifier. 
	"""
	for candi_cue, offset, __1, __2, __3 in candidate_cue_list:
		for real_cue, r_offset in real_cue_list:
			if offset == r_offset:
				# print zip(*candidate_cue_list)[0].index(candi_cue)
				real = candidate_cue_list.pop(zip(*candidate_cue_list)[0].index(candi_cue))
	return candidate_cue_list


def extractCandidates(mapping_connectives, amount=None):
	"""
	Find candidate connectives, if it signals discourse structure then it's a candidate.
	Extract n-grams to find phrases in the parsed data. 
	I want to see if the connective is already in the relation list, by searching on docid, sentid and possible connective
	the functions returns the negative samples in the dataset. 
	"""
	count = 0
	all_candidates = {}
	for docID in PARSES:
		sentences = PARSES[docID]['sentences']
		print '\n\n\n\n',docID
		print "amount of sentences: {}".format(len(sentences))
		doc_condidates = []
		
		for idx, sent in enumerate(sentences):
			sentence_number = idx
			tagged = [(el[0],el[1]['PartOfSpeech']) for el in sent['words']]
			tokens_and_offset = [(el[0], list([el[1]['CharacterOffsetBegin'],\
							 el[1]['CharacterOffsetEnd']])) for el in sent['words']]
			sentence = ' '.join(zip(*tokens_and_offset)[0])
			ngrams = ANALYZER(sentence)
			sorted_on_length = sorted(ngrams, key=lambda x:len(x))
			candidate_cues = comparePhrases(sorted_on_length, mapping_connectives)
			candidate_cues, removed = mapCandidates(tokens_and_offset, candidate_cues) #MAP THE WORDS TO THE OFFSET
			candidate_and_features = []
			if candidate_cues: #example: [(word, [[123,124]])]

				for element in candidate_cues:
					phrase_structure = get_phrase_structure(docID, sentence_number)
					splitted_cue_phrase = element[0].split()
					position = sentence.split().index(splitted_cue_phrase[0])
					# if len(splitted_cue_phrase)>1:
						# pos = None
					# else:
					dependency = get_connective_dependency(docID, sentence_number, element[0])[0]
					candidate_and_features.append((element[0], element[1], phrase_structure, position, dependency)) # [(word, [[123,124]], 'phrasestructure', position, dependency)]

		# 	#ALL THE CANDIDATES IN THE DOCUMENT
			real_connectives = returnRealConnectives(docID)
			doc_condidates.extend(candidate_and_features)

		negative_samples = returnNegativeSamples(real_connectives, doc_condidates)
		all_candidates[docID] = negative_samples
		count+=1
		if amount != None:
			if count==amount:
				return all_candidates
				break
	return all_candidates


def transformNeg(negative_samples):
	"""
	transform the dicitionairy into a list containing tuples with dictionaires inside
	"""
	negative_features = []
	for docID in negative_samples.keys():
		for neg_connective in negative_samples[docID]:
			negative_features.append(({'connective': neg_connective[0],\
			 'position': neg_connective[3], 'dependency':neg_connective[4], 'phrase_structure': neg_connective[2]},0))
	return negative_features


if __name__ == '__main__':
	print "Loading training data "
	KNOWN_RELATIONS = [json.loads(line) for line in open(ENGLISH_TRAIN+'relations.json', 'r')]
	print "Loading parsed data"
	PARSES = json.load(open(ENGLISH_TRAIN+'parses.json', 'r'))
	print 'DONE'
	# unique_conn_mapping = discourseConnectives()
	# negative_samples = extractCandidates(unique_conn_mapping) #stored this using pickle
	print "extracting features from the positive sampes"
	training_data = constructTrainingData(KNOWN_RELATIONS)
	print 'DONE'
	neg = pickle.load(open('v2_negative_samples.pickle', 'r')) #make sure the pickled data file is in the same directory
	neg_training = transformNeg(neg)
	print "extracting features from the positive sampes"
	training_data.extend(neg_training)
	train, target = zip(*training_data)
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, target, test_size=0.2)
	print 'DONE'
	print 'Training classifier'
	encoding = maxent.TypedMaxentFeatureEncoding.train(
	    zip(X_train, y_train), count_cutoff=3, alwayson_features=True)

	classifier = maxent.MaxentClassifier.train(
	    zip(X_train, y_train), bernoulli=False, encoding=encoding, trace=0)
	print 'DONE'

	print "Accuracy score: {}".format(accuracy(classifier, zip(X_test,y_test)))

