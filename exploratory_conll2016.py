import json
import codecs
import os

from sklearn.feature_extraction import DictVectorizer


ENGLISH_ = 'tutorial/conll16st-en-01-12-16-trial'


pdtb_file = codecs.open(ENGLISH_+'/relations.json', encoding='utf8')
relations = [json.loads(x) for x in pdtb_file]
parse_file = codecs.open('tutorial/conll16st-en-01-12-16-trial/parses.json', encoding='utf8')
en_parse_dict = json.load(parse_file)
en_example_relation = relations[10]
en_doc_id = en_example_relation['DocID']


some_features = []
for rel in relations:
	# some_features.append({rel['ID']:rel['Connective'], 'tokens_connective':rel['Connective']['TokenList']})
	print rel['Connective']['RawText']


# print "THIS IS A PARSETREE" 
# print en_parse_dict[en_doc_id]['sentences'][15]['parsetree'], '\n\n\n\n\n'

# print "THESE ARE THE DEPENDENCIES"
# print en_parse_dict[en_doc_id]['sentences'][15]['dependencies'], '\n\n\n\n\n'

# print "THESE ARE THE WORDS"
# print en_parse_dict[en_doc_id]['sentences'][15]['words'], '\n\n\n\n\n'




vec = DictVectorizer()
# X = vec.fit_transform(some_features)