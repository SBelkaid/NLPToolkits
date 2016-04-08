import json
import codecs
import os

from sklearn.feature_extraction import DictVectorizer


ENGLISH_TRIAL = 'tutorial/conll16st-en-01-12-16-trial'
ENGLISH_TRAIN = 'conll16st-en-zh-dev-train_LDC2016E50/conll16st-en-01-12-16-train'

#SET THE USE VARIABLE TO EITHER ENGLISH TRIAL OR THE ENGLISH TRAIN
USE = ENGLISH_TRAIN

#LOADING DATA
pdtb_file = codecs.open(USE+'/relations.json', encoding='utf8')
relations = [json.loads(x) for x in pdtb_file]
en_parse_dict = json.load(codecs.open(USE+'/parses.json', encoding='utf8'))
# en_example_relation = relations[10]
# en_doc_id = en_example_relation['DocID']



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


# 	# print connective, '\t\t',type_relation,'\t\t',sense_relation, '\t\t', arg1_raw
	# some_features.append({rel['ID']:rel['Connective'], 'tokens_connective':rel['Connective']['TokenList']})

	#SOME INFO THAT CAN BE VECTORIZED
	some_features.append({'id':rel_id, 'connective': connective, 'type': type_relation, 'arg1_raw':arg1_raw})


print some_features



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