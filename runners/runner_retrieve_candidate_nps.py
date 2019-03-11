import gzip
import pickle
from os.path import join, expanduser
import pandas as pd
import json

from article_analysis.parse import get_chunk, get_articles, find_doi_chunk_map, load_ngram_dist
import article_analysis.parse_ent as aape


head = -1
verbose = True
keywords = ['hypothesis', 'hypotheses', 'table']
batch_size = 50

fp_gen = expanduser('~/data/jstor/latest/')
fp_gen = '/home/valery/RE_Project/amj_ngrams/latest_listagg'

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

for dois_batch, j in zip(dois_batched, range(len(dois_batched))):
    if verbose:
        print('batch number {0}'.format(j))
    dch_dict = get_articles(dois_batch, registry_dict, input_path, prefix)
    for doi in dois_batch:
        ngram_order = 1
        article_ngrams = dch_dict[doi]
        ngram_positions_list = article_ngrams[ngram_order]

        ixs = []
        for keyword in keywords:
            if keyword in ngram_positions_list.keys():
                ixs += ngram_positions_list[keyword]
        print('doi {0}, len ixs {1}'.format(doi, len(ixs)))

        carticle = articles_ds[doi]
        chunks = aape.get_np_candidates(ixs, carticle, nlp, 1)
        total_counts, total_counts_raw, table, tree_dict = aape.choose_popular_np_phrases(chunks)
        df = pd.DataFrame(table, columns=['root', 'np', 'count'])
        df['doi'] = doi
        df_agg.append(df)

df0 = pd.concat(df_agg)

df0 = df0[['root', 'np', 'count', 'doi']].sort_values(['root', 'count'],
                                                      ascending=[True, False]).reset_index(drop=True)

df0.to_csv('{0}np_data.csv.gz'.format(output_path), compression='gzip')
