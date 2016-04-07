import json
import codecs
pdtb_file = codecs.open('conll16st-en-01-12-16-trial/relations.json', encoding='utf8')
relations = [json.loads(x) for x in pdtb_file];
example_relation = relations[10]
example_relation
