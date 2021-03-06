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
from sklearn.metrics import classification_report

# plt.style.use('ggplot')
ENGLISH_TRIAL = 'tutorial/conll16st-en-01-12-16-trial'
ENGLISH_TRAIN = 'conll16st-en-zh-dev-train_LDC2016E50/conll16st-en-01-12-16-train/'
PATH_RAW_DOCS = 'conll16st-en-zh-dev-train_LDC2016E50/conll16st-en-01-12-16-train/raw/'

#SET THE USE VARIABLE TO EITHER ENGLISH TRIAL OR THE ENGLISH TRAIN
USE = ENGLISH_TRAIN
VECTORIZER = CountVectorizer(ngram_range=(1,8), analyzer='word', lowercase=False)
ANALYZER = VECTORIZER.build_analyzer()


def get_phrase_structure(docID, sentenceID):
    """
    Retrieve phrase_structure from the parses.json file by filename and sentenceID.
    """
    sentencelist = PARSES[docID]["sentences"]
    phrase_structure = sentencelist[sentenceID]["parsetree"]
    return phrase_structure


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
	Compares the candidates generated by extract_negative_candidates to the real cue phrases in KNOWN_RELATIONS.
	If the cue is in the KNOWN_RELATIONS the it is removed from the candidate list. What is left over is
	used as the negative samples for the training data. 
	"""
	candidates = []
	for gram in sorted_ngrams:
		if gram in zip(*mapping_connectives)[0]:
			candidates.append(gram)
	return candidates


def return_pos(docID, sentence_number, token, offset):
	"""
	return pos tag given the docID sentence number, token and character offset
	"""
	# print PARSES[docID]['sentences'][sentence_number]['words']
	# [zip(*PARSES[docID]['sentences'][1]['words'])[0].index(token[0])][1]['PartOfSpeech']
	sentence = PARSES[docID]['sentences'][sentence_number]['words']
	if len(token.split()) > 1:
		combined = []
		for tok in token.split():
			PoS = sentence[zip(*sentence)[0].index(tok)][1]['PartOfSpeech']
			combined.append(PoS)
		return ' '.join(combined)
	else:
		return sentence[zip(*sentence)[0].index(token)][1]['PartOfSpeech']


def extract_candidates(mapping_connectives, amount=None):
	"""
	return negative samples from the parses file by comparing them with example connectives.
	Extract n-grams to find phrases in the parsed data. 
	I want to see if the connective is already in the relation list, by searching on docid, sentid and possible connective
	the functions returns the negative samples in the dataset. 
	"""
	count = 0
	all_candidates = {}
	for docID in PARSES:
		sentences = PARSES[docID]['sentences']
		# print '\n\n\n\n',docID
		# print "amount of sentences: {}".format(len(sentences))
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
			candidates_with_features = []
			if candidate_cues: #example: [(word, [[123,124]])]
				# print candidate_cues
	# 			print 'sentence id: {}, doc id: {}'.format(sentence_number, docID)

				for element in candidate_cues:
					phrase_structure = get_phrase_structure(docID, sentence_number)
					splitted_cue_phrase = element[0].split()
					position = sentence.split().index(splitted_cue_phrase[0])
					dependency = get_connective_dependency(docID, sentence_number, element[0])[0]
					PoS = return_pos(docID, sentence_number, element[0], element[1])
					candidates_with_features.append((element[0], element[1], phrase_structure, position, dependency, PoS)) # [(word, [[123,124]], 'phrasestructure', position, dependency, PoS)]


			doc_condidates.extend(candidates_with_features)

		all_candidates[docID] = doc_condidates
		count+=1
		if amount != None:
			if count==amount:
				return all_candidates
	return all_candidates


def change_format(samples):
	"""
	transform the dicitionairy into a list containing tuples with dictionaires inside
	"""
	features = []
	for docID in samples.keys():
		for sample in samples[docID]:
			features.append(({'connective': sample[0],
				'position': sample[3],
				'dependency':sample[4],
				'phrase_structure': sample[2],
				'PoS':sample[5]},0))
	return features


if __name__ == '__main__':
	print "Loading training data"
	KNOWN_RELATIONS = [json.loads(line) for line in open(ENGLISH_TRAIN+'relations.json', 'r')]
	print "Loading parsed data"
	PARSES = json.load(open(ENGLISH_TRAIN+'parses.json', 'r'))
	print 'DONE'
	known_connectives = discourseConnectives()
	print 'Extracting candidates'
	extracted_candidate_cues = extract_candidates(known_connectives)
	print 'DONE'
	extracted_candidate_features = zip(*change_format(extracted_candidate_cues))[0]
	print 'Loading classifier'
	clf = pickle.load(open('./CLASSIFIERS/binairy_classifier.classifier', 'r'))
	output = zip(extracted_candidate_features, clf.classify_many(extracted_candidate_features))

	# example_output = ({'PoS': u'CC',
	#    'connective': u'or',
	#    'dependency': u'indicative-10',
	#    'phrase_structure': u"( (S (NP (NP (DT The) (NNP London) (NN exchange) (POS 's)) (JJ electronic) (NN price-reporting) (NN system)) (VP (VBD provided) (NP (NP (ADJP (ADJP (RB only) (JJ indicative)) (, ,) (CC or) (JJ non-firm) (, ,)) (NNS prices)) (PP (IN for) (NP (QP (IN about) (CD 40)) (NNS minutes)))) (PP (IN on) (NP (NNP Manic))) (NP (NNP Monday))) (. .)) )\n",
	#    'position': 11},
	#   0)
	# first element of example_output is feature dict and second element is the attached label. 
	




