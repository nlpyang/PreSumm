import csv
import glob
import json
import random
import itertools


def clean(s):
    r = s.replace(' -RRB-', ')')
    r = r.replace('-LRB- ', '(')
    r = r.replace(' -RSB-', ']')
    r = r.replace('-LSB- ', '[')
    r = r.replace(' \'', '\'')
    r = r.replace(' `', '`')
    r = r.replace(' :', ':')
    r = r.replace(' ,', ',')
    r = r.replace(' .', '.')

    return r


def cut(s):
    sents = s.split('.')
    rtn = ''
    for s in sents:
        rtn += s + '.'
        if (len(rtn.split()) > 400):
            if (rtn[-1] != '.'):
                rtn += '.'
            return rtn
    if (rtn[-1] != '.'):
        rtn += '.'

    return rtn


# models = ['cnndm_oracle','cnndm_lead','cnndm_neusum','cnndm_classifier','cnndm_transformer','cnndm_gold']
models = ['bert', 'shashi', 'see', 'lead']

d = {}

for fp in glob.glob('../json_data/xsum2/*.test.*.json'):
    jobj = json.load(fp)
    print(len(jobj))
    # lines = open(fp).read().strip().split('\n')
    # lines = [clean(line).split('<q>') for line in lines]
    # d[fn] = lines

# qas = open('qa').read().strip().split('\n\n')
# qas = [qa.strip().split('\n') for qa in qas]
# with open('qa.csv', 'w') as csvfile:
#     fieldnames = ['model', 'summary', 'question-1', 'question-2']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#     for i in range(len(d[models[0]])):
#         candidates = d.keys()
#         for c in candidates:
#             to_write = {'model': c, 'summary': ' '.join(d[c][i]), 'question-1': qas[i][0].split('\t')[0],
#                         'question-2': qas[i][1].split('\t')[0]}
#             writer.writerow(to_write)
