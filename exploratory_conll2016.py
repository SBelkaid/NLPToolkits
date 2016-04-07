import json
import codecs
import os


ENGLISH_ = 'tutorial/conll16st-en-01-12-16-trial'
pdtb_file = codecs.open(ENGLISH_+'/relations.json', encoding='utf8')
relations = [json.loads(x) for x in pdtb_file]


parse_file = codecs.open('tutorial/conll16st-en-01-12-16-trial/parses.json', encoding='utf8')
en_parse_dict = json.load(parse_file)

en_example_relation = relations[10]
en_doc_id = en_example_relation['DocID']
# print en_parse_dict[en_doc_id]['sentences'][15]['parsetree']
print en_parse_dict[en_doc_id]['sentences'][15]['dependencies']

print en_parse_dict[en_doc_id]['sentences'][15]['words'][0]