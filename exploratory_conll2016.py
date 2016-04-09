import json
import codecs
import os
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


ENGLISH_TRIAL = 'tutorial/conll16st-en-01-12-16-trial'
ENGLISH_TRAIN = 'conll16st-en-zh-dev-train_LDC2016E50/conll16st-en-01-12-16-train'
PATH_RAW_DOCS = 'conll16st-en-zh-dev-train_LDC2016E50/conll16st-en-01-12-16-train/raw/'

#SET THE USE VARIABLE TO EITHER ENGLISH TRIAL OR THE ENGLISH TRAIN
USE = ENGLISH_TRAIN

#LOADING DATA
pdtb_file = codecs.open(USE+'/relations.json', encoding='utf8')
relations = [json.loads(x) for x in pdtb_file]
en_parse_dict = json.load(codecs.open(USE+'/parses.json', encoding='utf8'))
file_names = os.listdir(PATH_RAW_DOCS)
tfidf_all_files = [codecs.open(PATH_RAW_DOCS+f, encoding='latin-1').read() for f in file_names]
tfidf = TfidfVectorizer(min_df=1, encoding='latin-1', stop_words='english')
X = tfidf.fit_transform(tfidf_all_files)
# en_example_relation = relations[10]
# en_doc_id = en_example_relation['DocID']

def discourseConnectives():
	"""
	return a dictionairy of all the possible connective senses and how they occur
	by putting the words indicating the type in a list 
	"""
	connectives = defaultdict(set)
	for rel in relations:
		if rel['Type']=='Explicit':
				connectives[rel['Sense'][0]].add(rel['Connective']['RawText'].lower())
	return connectives

def calcSentSalience(sent):
	"""
	return list of tuples containing word and its tfidf value.
	Words that haven't been seen before when vectorizing all files are omitted.
	"""
	response = tfidf.transform([sent])
	feature_names = tfidf.get_feature_names()
	return [(feature_names[col], response[0, col]) for col in response.nonzero()[1]]


def onlyExplicit(max_amount=10, doc_id='wsj_0207'):
	"""
	this functions returns relations from documents with regards to a certain document id
	"""
	count = 0
	for rel in relations:
		if rel['Type']=='Explicit' and rel['DocID']==doc_id:
			if count < max_amount:
				print 'ARG1:', rel['Arg1']['RawText']
				print 'SALIENCE ARG1', sum(zip(*calcSentSalience(rel['Arg1']['RawText']))[1])
				print 'CONECTIVE:', rel['Connective']['RawText'].upper()
				print 'SENSE:',rel['Sense'][0]
				print 'ARG2:', rel['Arg2']['RawText']
				print 'SALIENCE ARG2', sum(zip(*calcSentSalience(rel['Arg2']['RawText']))[1]), '\n\n'
				count += 1


def extractData():
	"""
	extracts data from the given data structure by the task.
	"""

	all_features = []
	some_features = []
	for rel in relations:
		rel_id = rel['ID']
		#CONNECTIVE INFORMATION
		connective = rel['Connective']['RawText']
		#TYPE AND SENSE
		type_relation = rel['Type'] 
		sense_relation = rel['Sense'][0]
		#RAWTEXT BOTH ARGS
		arg1_raw = rel['Arg1']['RawText']
		arg2_raw = rel['Arg2']['RawText']
		#TOKENLIST FOR BOTH ARGS
		arg1_token_list = rel['Arg1']['TokenList']
		arg2_token_list = rel['Arg1']['TokenList']
		#AMOUNT OF ARGUMENTS FOR THE TOKENS
		arg1_amount_tokens = max(zip(*rel['Arg1']['TokenList'])[-1])
		arg2_amount_tokens = max(zip(*rel['Arg2']['TokenList'])[-1])
		#TAGGED TOKENS
		tags = []
		for sent in en_parse_dict[rel['DocID']]['sentences']:
		    words, rest = zip(*sent['words'])
		    pos_tags = [el['PartOfSpeech'] for el in rest]
		    # tags.append(pos_tags)
		    tags.append(zip(words, pos_tags))

		# zip(*en_parse_dict[rel['DocID']]['sentences'][0]['words'])[0]

	
	#ALL FEATURES SOME OF WHICH CAN'T BE VECTORIZED
	all_features.append({'id':rel_id, 'connective': connective, 'type': type_relation, 'arg1_raw':arg1_raw})


	# SOME INFO THAT CAN BE VECTORIZED
	some_features.append({'id':rel_id, 'connective': connective, 'type': type_relation, 'arg1_raw':arg1_raw})
	return some_features, all_features


# s, a = extractData()
onlyExplicit()
# print calcSentSalience('Solo woodwind players have to be creative if they want to work a lot, because their repertoire and audience appeal are limited.')





# print "THIS IS A PARSETREE" 
# print en_parse_dict[en_doc_id]['sentences'][15]['parsetree'], '\n\n\n\n\n'

# print "THESE ARE THE DEPENDENCIES"
# print en_parse_dict[en_doc_id]['sentences'][15]['dependencies'], '\n\n\n\n\n'

# print "THESE ARE THE WORDS"
# for el in en_parse_dict[en_doc_id]['sentences'][15]['words']:
# 	print el


#VECTORIZER ASKS FOR LIST OF DICTIONAIRY CONTAINING KEY AND VALUE. VALUE CAN'T BE LIST
# vec = DictVectorizer()
# X = vec.fit_transform(some_features)
# print vec.get_feature_names()