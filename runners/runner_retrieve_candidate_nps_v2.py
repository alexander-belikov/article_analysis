import gzip
import pickle
from os.path import join, expanduser
import pandas as pd
import json
import article_analysis.parse_ent as aape
from numpy import nan


head = 15
verbose = True
keywords = ['hypothesis', 'hypotheses', 'table']
batch_size = 5

fp_gen = expanduser('~/data/jstor/latest/')
# fp_gen = '/home/valery/RE_Project/amj_ngrams/latest_listagg/'

input_path = fp_gen
output_path = fp_gen
prefix = 'ngrams_dict'

nlp = aape.init_nlp()

with open(join(output_path, 'registry_json.txt')) as file:
    registry_dict = json.loads(file.read())

fname = '{0}corpus_clean_dict.pgz'.format(fp_gen)

with gzip.open(fname) as fp:
    articles_ds = pickle.load(fp)


all_dois_flat = [v for sublist in registry_dict.values() for v in sublist]

if head > 0:
    all_dois_flat = all_dois_flat[:head]

# check boundaries
dois_batched = [all_dois_flat[i:i + batch_size] for i in range(0, len(all_dois_flat), batch_size)]

df_agg = []

for j, dois_batch in enumerate(dois_batched):
    df_batch = []
    if verbose:
        print('batch number {0}'.format(j))
    for doi in dois_batch:
        if verbose:
            print('doi: {0}'.format(doi))

        carticle = articles_ds[doi]
        dfr = aape.combine_keyword_table(carticle, keywords, nlp, 0.4, 10)
        if dfr.shape[0] == 0:
            dfr = pd.DataFrame([[nan] * dfr.shape[1]], columns=dfr.columns)
        dfr['doi'] = doi
        n_phrases = len(carticle)
        dfr['n_phrases'] = n_phrases
        n_words = sum([len(x) for x in carticle])
        dfr['n_words'] = n_words
        c_sorted = sorted(dfr.columns, key=lambda x: len(x))
        df_agg.append(dfr[c_sorted])
        df_batch.append(dfr[c_sorted])

    df_tmp = pd.concat(df_batch)
    df_tmp.to_csv('{0}np_stats_batch_{1}.csv.gz'.format(output_path, j), compression='gzip')

df0 = pd.concat(df_agg)
#
# df0 = df0[['root', 'np', 'count', 'doi']].sort_values(['root', 'count'],
#                                                       ascending=[True, False]).reset_index(drop=True)
#
df0.to_csv('{0}np_stats.csv.gz'.format(output_path), compression='gzip')
