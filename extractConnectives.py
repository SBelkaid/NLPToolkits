import json
import codecs
import os
# import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.classify import maxent
from nltk.classify.util import accuracy
from sklearn.feature_extraction.text import CountVectorizer
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
		dependency_heading, dependency_attached = get_connective_dependency(docID, sentenceID, part_of_speech, connective)
		connectivelist.append(connective.lower())
		features = {'connective': connective,
		 'pos': part_of_speech,
		 'position': sentence_position,
		 'dependency':dependency_heading,
		 'phrase_structure':phrase_structure}
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
		tuplelist.append((features, 1))
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

def mapCandidates(tokens, candidates):
	"""
	iterates through the tokens and offset parameter and finds the offset of the possible candidate.
	This has been done because after making the ngrams the original token data is lost and requires
	mapping. First the candidate cues are sorted based on length of phrase, this is done because 
	an 1-gram can refer to the same word in an ngram so they can't be seen as seperate. Therefore,
	the longer cues are extracted first. 
	"""
	identified_candidates = []
	for candi in sorted(candidates, key=lambda x: len(x.split()), reverse=True):
		candi_tokens = candi.split()
		if len(candi_tokens) > 1:
			phrase = []
			for word in candi_tokens:
				if word in zip(*tokens)[0]:
					# print tokens[zip(*tokens)[0].index(word)]
					phrase.append(tokens[zip(*tokens)[0].index(word)])
					tokens.remove(tokens[zip(*tokens)[0].index(word)])
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
	return identified_candidates

def comparePhrases(sorted_ngrams, mapping_connectives):
	candidates = []
	for gram in sorted_ngrams:
		if gram in zip(*mapping_connectives)[0]:
			candidates.append(gram)
	return candidates

def showConnectives(doc_ID):
	"""
	print all the connectives in a document.
	"""
	real_connectives = []
	for rel in KNOWN_RELATIONS:
		if rel['DocID']== doc_ID and rel['Type'] == 'Explicit':
			real_connectives.append((rel['Connective']['RawText'],rel['Connective']['CharacterSpanList']))
			# print rel['Connective']['RawText'], rel['Connective']['CharacterSpanList']
	return real_connectives

def extractConstituentStruct(candidate_cue, docID):
	"""
	Extract constituent structure given the candidate connective, the document id and the Span offset which is with
	the candidate cue 
	"""
	pass

def extractCandidates(mapping_connectives, amount=20):
	"""
	Find candidate connectives, if it signals discourse structure then it's a candidate.
	Extract n-grams to find phrases in the parsed data. 
	I want to see if the connective is already in the relation list, by searching on docid, sentid and possible connective
	"""
	count = 0
	for docID in PARSES:
		sentences = PARSES[docID]['sentences']
		print '\n\n\n\n',docID
		print "amount of sentences: {}".format(len(sentences))
		candidates = []
		for idx, sent in enumerate(sentences):
			tagged = [(el[0],el[1]['PartOfSpeech']) for el in sent['words']]
			tokens_and_offset = [(el[0], list([el[1]['CharacterOffsetBegin'], el[1]['CharacterOffsetEnd']])) for el in sent['words']]
			sentence = ' '.join(zip(*tokens_and_offset)[0])
			ngrams = ANALYZER(sentence)
			sorted_on_length = sorted(ngrams, key=lambda x:len(x))
			candidate_cues = comparePhrases(sorted_on_length, mapping_connectives)
			candidate_cues = mapCandidates(tokens_and_offset, candidate_cues) #MAP THE WORDS TO THE OFFSET
			#ALL THE CANDIDATES IN THE DOCUMENT
			candidates.extend(candidate_cues)
			real_connectives = showConnectives(docID)

		print candidates
		print real_connectives
		count+=1
		if count==amount:
			break
		

if __name__ == '__main__':
	KNOWN_RELATIONS = [json.loads(line) for line in open(ENGLISH_TRAIN+'relations.json', 'r')]
	PARSES = json.load(open(ENGLISH_TRAIN+'parses.json', 'r'))
	# unique_conn_mapping = discourseConnectives()
	# extractCandidates(unique_conn_mapping)
	doc_id = 'wsj_0802'
	candi = (u'for', [[84, 87]])
	extractConstituentStruct(candi,docID)
	# training_data = constructTrainingData(KNOWN_RELATIONS)

	# test =  [
	#     {'connective': 'until', 'pos': 'IN', 'position': 18},
	#     {'connective': 'until', 'pos': 'IN', 'position': 6},
	#     {'connective': 'before', 'pos': 'IN', 'position': 13},
	#     {'connective': 'Moreover', 'pos': 'RB', 'position': 0},
	#     {'connective': 'but', 'pos': 'CC', 'position': 16},
	#     {'connective': 'after', 'pos': 'IN', 'position': 21},
	#     {'connective': 'Testing', 'pos': 'XXX', 'position': 90}
	#   ]

	# encoding = maxent.TypedMaxentFeatureEncoding.train(
	#     training_data, count_cutoff=3, alwayson_features=True)

	# classifier = maxent.MaxentClassifier.train(
	#     training_data, bernoulli=False, encoding=encoding, trace=0)

	# # print classifier.classify_many(test)

