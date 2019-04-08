import spacy
import networkx as nx
import pandas as pd
import numpy as np
from itertools import chain, combinations
from functools import reduce


def init_nlp():
    nlp = spacy.load('en')
    not_stop_words = ['top', 'bottom']
    stop_words = ["'s", '-', '.', "'", '"']

    for w in not_stop_words:
        #     nlp.Defaults.stop_words.remove(w)
        nlp.vocab[w].is_stop = False

    for w in stop_words:
        nlp.vocab[w].is_stop = True

    return nlp


def get_np_candidates(iphrases, article, nlp, window=1, graph_mode=True):
    """

    :param iphrases: indices of phrases of interest in the article
    :param article: list of phrases
    :param nlp: spacy nlp module
    :param window:
    :param graph_mode:
    :return: list of reduced noun phrase candidates lemmatized (root, phrase, phrase-Tree)
    """

    chunks = []
    inds = [range(i - window, i + window + 1) for i in iphrases]
    overlap_indices = sorted(list(set([x for sublist in inds for x in sublist])))
    overlap_indices = [ii for ii in overlap_indices if ii < len(article)]

    for j in overlap_indices:
        phrase = ' '.join(article[j])
        doc = nlp(phrase)
        for chunk in doc.noun_chunks:
            supp_chunk = [c for c in chunk if not (c.is_stop or c.like_num or c.is_punct
                                                   or c.text.lower() in nlp.Defaults.stop_words)]
            if supp_chunk and chunk.root.pos_ == 'NOUN':

                supp_chunk2 = [c.lemma_ for c in supp_chunk]
                if graph_mode:
                    edge_slist = [[(c.lemma_, d.lemma_) for d in c.children if d in supp_chunk] for c in supp_chunk]
                    edge_list = [e for sublist in edge_slist for e in sublist]
                    g = nx.DiGraph()
                    edge_list += [('#', chunk.root.lemma_)]
                    g.add_edges_from(edge_list)
                    chunks.append((chunk.root.lemma_, tuple(supp_chunk2), g))
                else:
                    chunks.append((chunk.root.lemma_, tuple(supp_chunk2), j))
    return chunks


def choose_popular_np_phrases(candidates, thr_root_length=2, thr_root_popularity=5, thr_np_length=1,
                              thr_np_popularity=1):
    root_dict = {}
    tree_dict = {}
    raw_dict = {}

    for root, np, tr in candidates:
        if len(np) > thr_np_length and len(root) > thr_root_length:
            if np in raw_dict.keys():
                raw_dict[np] += 1
            else:
                raw_dict[np] = 1
            if root in root_dict:
                if np in root_dict[root].keys():
                    root_dict[root][np] += 1
                else:
                    root_dict[root][np] = 1
                    tree_dict[root][np] = tr
            else:
                root_dict[root] = {np: 1}
                tree_dict[root] = {np: tr}

    # total_counts = {k: sum(root_dict[k].values()) for k in list(root_dict.keys())}
    total_counts = sorted([(k, sum(v.values())) for k, v in root_dict.items()], key=lambda x: -x[1])
    total_counts_raw = sorted([(k, v) for k, v in raw_dict.items()], key=lambda x: -x[1])
    total_counts_raw_dict = dict(total_counts_raw)

    root_candidates = [k for k, v in total_counts
                       if v > thr_root_popularity and len(k) > thr_root_length]
    red_tree_dict = {k: tree_dict[k] for k, v in tree_dict.items() if k in root_candidates}

    table_ = [[(root, np, total_counts_raw_dict[np]) for np in root_dict[root].keys()
               if total_counts_raw_dict[np] > thr_np_popularity] for root in root_dict.keys()]
    table = [x for sublist in table_ for x in sublist]
    return total_counts, total_counts_raw, table, red_tree_dict


def find_tablelike_phrases(article, thr_freq=0.4, thr_len=10,
                           thr_table_distance_normed=0.05, verbose=False):
    stat = pd.DataFrame([(len([x for x in phrase if x.isdigit()]),
                          len(phrase),
                          'table' in phrase or 'Table' in phrase) for phrase in article],
                        columns=['n_digitlike', 'n', 'table_flag'])
    stat['freq'] = stat['n_digitlike'] / stat['n']
    #     distance to table : 0 for very far, 1 in the same phrase
    stat = stat.reset_index()
    iis = list(stat.loc[stat['table_flag'], 'index'].index)
    article_length = len(article)
    if iis:
        stat['table_distance'] = stat['index'].apply(lambda x: np.abs(min([i - x for i in iis])))
    else:
        stat['table_distance'] = article_length

    stat['table_distance_normed'] = stat['table_distance'].apply(lambda x: x / article_length)
    stat['table_flag_ext'] = stat['table_flag'] | stat['table_flag'].shift()
    mask = ((stat['freq'] > thr_freq) &
            (stat['n'] > thr_len) &
            (stat['table_distance_normed'] < thr_table_distance_normed))
    if verbose:
        print(sum(stat['table_flag']), sum(mask))
    iis = stat.loc[mask, 'index'].values
    return iis


def powerset(iterable, nonempty=True):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    nonempty = int(nonempty)
    return chain.from_iterable(combinations(s, r) for r in range(nonempty, len(s)+1))


def get_root0_stats(carticle, keys, nlp, np_bag=None, phrase_grams_dict=None, powerset_flag=True,
                    shift=0, second_order=False, verbose=False):
    """

    :param carticle:
    :param keys:
    :param nlp:
    :param np_bag:
    :param phrase_grams_dict:
    :param powerset_flag:
    :param shift:
    :param verbose:
    :param second_order: if true keys is a dict {root: noun phrase]}
    :return:
    """
    # bag of np from a list of phrases
    n = len(carticle)
    if not np_bag:
        np_bag = get_np_candidates(range(len(carticle)), carticle, nlp, 0, graph_mode=False)

    if not phrase_grams_dict:
        phrase_grams_dict = {}

        # dict : {root: {noun phrase: [occurences]}}
        for root, chunk, pos in np_bag:
            if root in phrase_grams_dict.keys():
                if chunk in phrase_grams_dict[root].keys():
                    phrase_grams_dict[root][chunk] += [pos]
                else:
                    phrase_grams_dict[root][chunk] = [pos]
            else:
                phrase_grams_dict[root] = {chunk: [pos]}
    if verbose:
        print('len phrase gram dict {0}'.format(len(phrase_grams_dict)))

    if second_order:
        root_ix_dict = {root: list(set.union(*[set(ixs) for nphrase, ixs
                                               in phrase_grams_dict[root].items() if nphrase in nphrases]))
                        if root in phrase_grams_dict.keys() else [] for root, nphrases in keys.items()}
    else:
        # for readability {root: {noun phrase: [occurrences]}} -> {root: [occurrences]} for keywords
        root_ix_dict = {key: list(set.union(*[set(x) for x in phrase_grams_dict[key].values()]))
                        if key in phrase_grams_dict.keys() else [] for key in keys}

    root_ix_dict = {k: set([x+shift for x in v if (x+shift > -1 or x+shift < n)]) for k, v in root_ix_dict.items()}

    vcs = []
    roots_dict_agg = []

    if second_order:
        ix_phrases = list(set.union(*[set(x) for x in root_ix_dict.values()]))
        vc, roots_dict = get_vc_nps_from_phrases(np_bag, ix_phrases)
        cnt_col_name = '{0}_{1}_shift{2}'.format('cnt', 'second_order', shift)
        vc = pd.DataFrame(vc, columns=[cnt_col_name])
        vcs += [vc]
        roots_dict_agg += [roots_dict]
    else:
        if powerset_flag:
            keys_gen = powerset(keys)
        else:
            keys_gen = keys

        for key_tuple in keys_gen:
            if verbose:
                print('key_tuple: {0}'.format(key_tuple))
            # select roots of interest
            ix_phrases = [root_ix_dict[k] for k in key_tuple if k in root_ix_dict]
            if verbose:
                print('ix_phrases: {0}'.format(ix_phrases))
            ix_phrases = list(set.union(*ix_phrases))
            vc, roots_dict = get_vc_nps_from_phrases(np_bag, ix_phrases)
            cnt_col_name = '{0}_{1}_shift{2}'.format('cnt', '_'.join(key_tuple), shift)
            vc = pd.DataFrame(vc, columns=[cnt_col_name])
            vcs += [vc]
            roots_dict_agg += [roots_dict]

    roots_dict_inv = {vv: k for k, v in roots_dict.items() for vv in v}
    if verbose:
        print('len vcs {0}'.format(len(vcs)))
    vcs = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='outer'), vcs)
    vcs['first_occ'] = vcs.index.to_series().apply(lambda x: np.min(phrase_grams_dict[roots_dict_inv[x]][x])/n)
    vcs = vcs.fillna(0)
    return vcs, roots_dict_agg, np_bag, phrase_grams_dict


def get_vc_nps_from_phrases(np_bag, ix_phrases):
    if ix_phrases:
        # (root, np, index) from sentences containing the roots of interest
        chunks = [item for item in np_bag if item[2] in ix_phrases]

        roots_dict = {}
        for root, nphrase, _ in chunks:
            if root in roots_dict:
                roots_dict[root] |= {nphrase}
            else:
                roots_dict[root] = {nphrase}
        # value_counts of chunks
        vc = pd.Series([chunk for _, chunk, _ in chunks]).value_counts()
    else:
        roots_dict = {}
        vc = pd.Series()
    return vc, roots_dict


def collect_full_stats(carticle, keywords, nlp):
    np_bag = None
    np_dict = None
    shifts = [0, 1]
    shifted_vcs = []
    shifted_roots = []
    for shift in shifts:
        vcs, roots, np_bag, np_dict = get_root0_stats(carticle, keywords, nlp, np_bag,
                                                      np_dict, shift=shift)
        shifted_vcs.append(vcs)
        shifted_roots.append(roots)

    tot_vcs = reduce(lambda x, y: x.merge(y[list(set(y.columns) - {'first_occ'})],
                                          left_index=True, right_index=True, how='outer'),
                     shifted_vcs)

    for vc in shifted_vcs[1:]:
        tot_vcs.update(vc['first_occ'])
    tot_vcs = tot_vcs.fillna(0)

    dd_agg = {}
    for list_dicts in shifted_roots:
        for plain_dict in list_dicts:
            for k, v in plain_dict.items():
                if k in dd_agg.keys():
                    dd_agg[k] |= set(v)
                else:
                    dd_agg[k] = set(v)

    vcs_second, roots, np_bag, np_dict = get_root0_stats(carticle, dd_agg, nlp, np_bag,
                                                         np_dict, second_order=True, verbose=True)

    dfr = pd.merge(vcs_second, tot_vcs[list(set(tot_vcs.columns) - {'first_occ'})],
                   how='outer', left_index=True, right_index=True)
    dfr.update(tot_vcs['first_occ'])
    dfr = dfr.fillna(0)

    return dfr

