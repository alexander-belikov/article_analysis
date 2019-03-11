import gzip
import pickle
from os.path import join, expanduser
import pandas as pd
import json

from article_analysis.parse import get_chunk, get_articles, find_doi_chunk_map, load_ngram_dist
import article_analysis.parse_ent as aape


head = 50
verbose = True

nlp = aape.init_nlp()

input_path = expanduser('~/data/jstor/latest/')
output_path = expanduser('~/data/jstor/latest/')
prefix = 'ngrams_dict'

with open(join(output_path, 'registry_json.txt')) as file:
    registry_dict = json.loads(file.read())


fname = expanduser('~/data/jstor/latest/corpus_clean_dict.pgz')
with gzip.open(fname) as fp:
    articles_ds = pickle.load(fp)


all_dois_flat = [v for sublist in registry_dict.values() for v in sublist]

if head > 0:
    all_dois_flat = all_dois_flat[:50]

# check boundaries
batch_size = 10
dois_batched = [all_dois_flat[i:i+batch_size] for i in range(0, len(all_dois_flat), batch_size)]

df_agg = []

for dois_batch, j in zip(dois_batched, range(len(dois_batched))):
    if verbose:
        print('batch number {0}'.format(j))
    dch_dict = get_articles(dois_batch, registry_dict, input_path, prefix)
    for doi in dois_batch:
        ngram_order = 1
        article_ngrams = dch_dict[doi]
        ngram_positions_list = article_ngrams[ngram_order]
        ixs = ngram_positions_list['hypothesis'] + ngram_positions_list['table']

        carticle = articles_ds[doi]
        chunks = aape.get_np_candidates(ixs, carticle, nlp, 1)
        total_counts, total_counts_raw, table, tree_dict = aape.choose_popular_np_phrases(chunks)
        df = pd.DataFrame(table, columns=['root', 'np', 'count'])
        df_agg.append(df)

