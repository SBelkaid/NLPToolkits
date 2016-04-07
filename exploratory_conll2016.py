import json
import codecs
import os


ENGLISH_ = 'tutorial/conll16st-en-01-12-16-trial'
pdtb_file = codecs.open(ENGLISH_+'/relations.json', encoding='utf8')
relations = [json.loads(x) for x in pdtb_file]
example_relation = relations[10]
print example_relation
