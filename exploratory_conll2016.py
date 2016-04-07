import json
import codecs
import os

from sklearn.feature_extraction import DictVectorizer


ENGLISH_TRIAL = 'tutorial/conll16st-en-01-12-16-trial'
ENGLISH_TRAIN = 'conll16st-en-zh-dev-train_LDC2016E50/conll16st-en-01-12-16-train'

#SET THE USE VARIABLE TO EITHER ENGLISH TRIAL OR THE ENGLISH TRAIN
USE = ENGLISH_TRAIN


pdtb_file = codecs.open(USE+'/relations.json', encoding='utf8')
relations = [json.loads(x) for x in pdtb_file]
parse_file = codecs.open(USE+'/parses.json', encoding='utf8')
en_parse_dict = json.load(parse_file)
en_example_relation = relations[10]
en_doc_id = en_example_relation['DocID']


some_features = []
for rel in relations:
	rel_id = rel['ID']
	connective = rel['Connective']['RawText']
	type_relation = rel['Type'] 
	sense_relation = rel['Sense'][0]
	arg1_raw = rel['Arg1']['RawText']
	arg1_token_list = rel['Arg1']['TokenList']
	# print connective, '\t\t',type_relation,'\t\t',sense_relation, '\t\t', arg1_raw
	# some_features.append({rel['ID']:rel['Connective'], 'tokens_connective':rel['Connective']['TokenList']})
	some_features.append({'id':rel_id, 'connective': connective, 'type': type_relation, 'arg1_raw':arg1_raw})
	# 'arg1_token_list':arg1_token_list


# print "THIS IS A PARSETREE" 
# print en_parse_dict[en_doc_id]['sentences'][15]['parsetree'], '\n\n\n\n\n'

# print "THESE ARE THE DEPENDENCIES"
# print en_parse_dict[en_doc_id]['sentences'][15]['dependencies'], '\n\n\n\n\n'

# print "THESE ARE THE WORDS"
# print en_parse_dict[en_doc_id]['sentences'][15]['words'], '\n\n\n\n\n'




vec = DictVectorizer()
X = vec.fit_transform(some_features)
print vec.get_feature_names()