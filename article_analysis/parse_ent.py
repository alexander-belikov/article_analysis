import spacy
import networkx as nx
import pandas as pd
import numpy as np
from itertools import chain, combinations, product
from functools import reduce
from operator import and_


def init_nlp():
    nlp = spacy.load('en')
    not_stop_words = ['top', 'bottom', 'of']
    stop_words = ["'s", '-', '.', "'", '"']

    for w in not_stop_words:
        #     nlp.Defaults.stop_words.remove(w)
        nlp.vocab[w].is_stop = False

    for w in stop_words:
        nlp.vocab[w].is_stop = True

    return nlp


def get_np_candidates(iphrases, article, nlp, window=1, graph_mode=True):
    """
    from indices of phrases iphrases from article collect lemmatized noun phrases
        as a list of (root, phrase, phrase-Tree)

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


def powerset(iterable, nonempty=True):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    :param iterable:
    :param nonempty:
    :return:
    """

    s = list(iterable)
    nonempty = int(nonempty)
    return chain.from_iterable(combinations(s, r) for r in range(nonempty, len(s)+1))


def get_keyword_phrase_ixs(carticle, keys, nlp, np_bag=None, foos=None, keys_flag=None,
                           shift=0, second_order=False, root_flag=True, verbose=False):
    """
    gather on noun phrases

    :param carticle:
    :param keys:
    :param nlp:
    :param np_bag:
    :param phrase_grams_dict:
    :param keys_flag: None, 'powerset', 'or'
    :param shift:
    :param verbose:
    :param second_order: if true keys is a dict {root: noun phrase]}
    :return:
    """
    # bag of np from a list of phrases
    n = len(carticle)
    if not np_bag:
        np_bag = get_np_candidates(range(len(carticle)), carticle, nlp, 0, graph_mode=False)

    # derive foos : root2chunk, chunk2root, pos2chunk, chunk2pos, pos2root, root2pos if needed
    if not foos:
        root2chunk, chunk2root, pos2chunk, chunk2pos, pos2root, root2pos = \
            dict(), dict(), dict(), dict(), dict(), dict()
        for root, chunk, pos in np_bag:
            if root in root2chunk.keys():
                root2chunk[root] |= {chunk}
            else:
                root2chunk[root] = {chunk}
            if root in root2pos.keys():
                root2pos[root] |= {pos}
            else:
                root2pos[root] = {pos}

            if pos in pos2root.keys():
                pos2root[pos] |= {root}
            else:
                pos2root[pos] = {root}

            if pos in pos2chunk.keys():
                pos2chunk[pos] |= {chunk}
            else:
                pos2chunk[pos] = {chunk}

            if chunk in chunk2pos.keys():
                chunk2pos[chunk] |= {pos}
            else:
                chunk2pos[chunk] = {pos}

            if chunk in chunk2root.keys():
                chunk2root[chunk] |= {root}
            else:
                chunk2root[chunk] = {root}
        foos = root2chunk, chunk2root, pos2chunk, chunk2pos, pos2root, root2pos

    else:
        root2chunk, chunk2root, pos2chunk, chunk2pos, pos2root, root2pos = foos
    # if verbose:
    #     print('len phrase gram dict {0}'.format(len(phrase_grams_dict)))
    combi_ix_dict = {}

    if root_flag:
        lemma_ix_dict = {k: root2pos[k] if k in root2pos.keys() else set() for k in keys}
    else:
        lemma_ix_dict = {k: chunk2pos[k] if k in chunk2pos.keys() else set() for k in keys}

    if not second_order:
        lemma_ix_dict = {k: set([x + shift for x in list(v) if (-1 < x + shift < n)]) for k, v in lemma_ix_dict.items()}

        if keys_flag == 'powerset':
            keys_gen = powerset(keys)
        elif keys_flag == 'or':
            keys_gen = [tuple(sorted(set.union((set(keys)))))]
        else:
            keys_gen = keys

        for key_tuple in keys_gen:
            if verbose:
                print('key_tuple: {0}'.format(key_tuple))
            # select roots of interest
            ix_phrases = [lemma_ix_dict[k] for k in key_tuple if k in lemma_ix_dict]

            if verbose:
                print('ix_phrases: {0} {1}'.format(key_tuple, ix_phrases))
            if ix_phrases:
                ix_phrases = sorted(set.union(*ix_phrases))
            else:
                ix_phrases = []

            if len(keys) > 5:
                combi_ix_dict['second_order'] = ix_phrases
            else:
                combi_ix_dict[key_tuple] = ix_phrases
            return combi_ix_dict, np_bag, foos
    else:
        # in case of second order : identify the first order words
        lv = list(lemma_ix_dict.values())
        iis = sorted(set.union(*lv))
        candidate_nps = [pos2chunk[j] for j in iis if j in pos2chunk.keys()]
        if candidate_nps:
            target_nps = sorted(set.union(*candidate_nps))
        else:
            target_nps = []
        # and run the first order wit shift 0 and root_flag False on them
        return get_keyword_phrase_ixs(carticle, target_nps, nlp, np_bag, foos, keys_flag='or',
                                      shift=0, second_order=False, root_flag=False, verbose=verbose)


def apply_vector_set_operators(set_vector, operator_vector):
    def ig_right(a, _):
        return a
    superset = set.union(*set_vector)

    def andnot(a, b):
        return and_(a, superset - b)
    conversion_dict = {'ig': ig_right, 'and': and_, 'andnot': andnot}
    if len(set(operator_vector) - set(conversion_dict.keys())) > 0:
        return set()
    else:
        operator_vector = [conversion_dict[x] for x in operator_vector] + [ig_right]
        set_vector = [superset] + set_vector
        combo = list(zip(set_vector, operator_vector))
        r_set, _ = reduce(lambda a, b: (a[1](a[0], b[0]), b[1]), combo)
        return r_set


def get_stats_for_ix_phrases(foos, ix_phrases):
    root2chunk, chunk2root, pos2chunk, chunk2pos, pos2root, root2pos = foos
    if ix_phrases:
        chunks = [item for ix in sorted(ix_phrases) if ix in pos2chunk.keys() for item in pos2chunk[ix]]
        s = pd.Series(chunks, name='np_count')
        vc = s.value_counts()
        df = vc.reset_index().rename(columns={'index': 'np'})
        df['root'] = df['np'].apply(lambda x: tuple(chunk2root[x])[0])
        return df
    else:
        return pd.DataFrame(columns=['np', 'root', 'np_count'])


def get_stats_for_set_vectors_ix(vector_ix_sets, matrix_ops, foos, case_names=None, verbose=False):
    root2chunk, chunk2root, pos2chunk, chunk2pos, pos2root, root2pos = foos

    if not case_names:
        case_names = ['_'.join(m) for m in matrix_ops]
    vcs = []
    for op_vec, suffix in zip(matrix_ops, case_names):
        ixs_result = apply_vector_set_operators(vector_ix_sets, op_vec)
        if verbose:
            print(ixs_result)
        df = get_stats_for_ix_phrases(foos, ixs_result)
        df = df.rename(columns={'np_count': 'np_count.{0}'.format(suffix)})
        if verbose:
            print(df.shape)
        vcs.append(df)
    n = max(pos2root.keys())
    vcs = reduce(lambda x, y: pd.merge(x, y, left_on=['np', 'root'], right_on=['np', 'root'], how='outer'), vcs)
    vcs['np_first_occ'] = vcs['np'].apply(lambda x: np.min(list(chunk2pos[x]))/n)
    vcs['root_first_occ'] = vcs['root'].apply(lambda x: np.min(list(root2pos[x]))/n)
    vcs = vcs.fillna(0)
    return vcs


def get_high_level_stats(carticle, keywords, nlp):
    np_bag = None
    foos = None
    keyword_ix_dict, np_bag, foos = get_keyword_phrase_ixs(carticle, keywords, nlp, np_bag, foos,
                                                           keys_flag='or', second_order=False, shift=0)

    keyword_ix_dict2, np_bag, foos = get_keyword_phrase_ixs(carticle, keywords, nlp, np_bag, foos,
                                                            keys_flag='or', second_order=False, shift=1)

    keyword_ix_dict_so, np_bag, foos = get_keyword_phrase_ixs(carticle, keywords, nlp, np_bag, foos,
                                                              keys_flag='or', second_order=True,
                                                              shift=0, root_flag=True,
                                                              verbose=False)

    op_vec = ['and', 'ig', 'ig']
    op_vec2 = ['andnot', 'and', 'ig']
    op_vec3 = ['andnot', 'andnot', 'and']

    dicts_ix = [keyword_ix_dict, keyword_ix_dict2, keyword_ix_dict_so]
    vector_ixs = [set(x[list(x.keys())[0]]) for x in dicts_ix]
    matrix_ops = [op_vec, op_vec2, op_vec3]

    vc0 = get_stats_for_set_vectors_ix(vector_ixs, matrix_ops, foos)

    return vc0, foos


# functions for dealing with tables

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


def parse_table_phrase(phrase):
    phrase_agg = []
    current = []
    for p in phrase:
        if p.isdigit() or p in ['-', '.', ':', '+']:
            phrase_agg.append(current)
            current = []
        else:
            current.append(p)
    phrase_agg = [p for p in phrase_agg if p]
    return phrase_agg


def parse_table_phrases(carticle, iis, nlp):
    np_bags = []
    for ix in iis:
        phrases = parse_table_phrase(carticle[ix])
        np_bag = get_np_candidates(range(len(phrases)), phrases, nlp, 0, graph_mode=False)
        np_bag = [(chunk, root, ix) for chunk, root, _ in np_bag]
        np_bags.extend(np_bag)
    return np_bags


def get_high_level_stats_table(carticle, nlp, foos, thr_freq=0.4, thr_len=10):
    root2chunk, chunk2root, pos2chunk, chunk2pos, pos2root, root2pos = foos
    # find indices of table-like phrases
    iis = find_tablelike_phrases(carticle, thr_freq=thr_freq, thr_len=thr_len)
    # get noun phrases from them, after cutting them by stop-words
    npb = parse_table_phrases(carticle, iis, nlp)

    # np is called np_count for later convenience
    df = pd.DataFrame(npb, columns=['root', 'np_count', 'pos'])

    # chunk to position
    np_min_pos = dict(df.groupby(['np_count']).apply(lambda x: x.pos.min()))
    # root to position
    root_min_pos = dict(df.groupby(['root']).apply(lambda x: x.pos.min()))
    # chunk to root, just in case the full article chunks are different
    chunk2root_table = dict(df.set_index(['np_count'])['root'].iteritems())

    # value counts of np
    vc = pd.DataFrame(df['np_count'].value_counts()).reset_index().rename(columns={'index': 'np'})

    n = max(pos2root.keys())
    vc['root'] = vc['np'].apply(lambda x: chunk2root_table[x])
    vc['root_first_occ'] = vc['root'].apply(
        lambda x: min(list(root2pos[x])) / n if x in root2pos.keys() else root_min_pos[x] / n)
    vc['np_first_occ'] = vc['np'].apply(
        lambda x: min(list(chunk2pos[x])) / n if x in chunk2pos.keys() else np_min_pos[x] / n)
    return vc


def combine_keyword_table(carticle, keywords, nlp, thr_freq, thr_len):
    vc0_, foos = get_high_level_stats(carticle, keywords, nlp)
    vc_table = get_high_level_stats_table(carticle, nlp, foos, thr_freq, thr_len)
    vc_table = vc_table.rename(columns={'np_count': 'np_count.{0}'.format('table')})
    non_occ_columns = [c for c in vc_table.columns if 'occ' not in c]
    vcs = [vc0_, vc_table[non_occ_columns]]
    vcs = reduce(lambda x, y: pd.merge(x, y, left_on=['np', 'root'], right_on=['np', 'root'], how='outer'), vcs)
    vcs['np_count.{0}'.format('table')] = vcs['np_count.{0}'.format('table')].fillna(0)

    null_mask = vcs['np_first_occ'].isnull()
    extra_nps = vcs.loc[null_mask, 'np'].tolist()

    mask_extras = vc_table['np'].isin(extra_nps)
    np_occ_dict = dict(vc_table.loc[mask_extras, ['np', 'np_first_occ']].values)
    root_occ_dict = dict(vc_table.loc[mask_extras, ['root', 'root_first_occ']].values)

    vcs.loc[null_mask, 'np_first_occ'] = vcs.loc[null_mask,
                                                 ['np', 'np_first_occ']].apply(lambda x: np_occ_dict[x['np']], axis=1)
    vcs.loc[null_mask, 'root_first_occ'] = vcs.loc[null_mask,
                                                   ['root', 'root_first_occ']].apply(lambda x: root_occ_dict[x['root']],
                                                                                   axis=1)
    return vcs

# leftover

def get_vcs_from_keyword_ix_dict(list_key_ix_phrases_dict, identifiers, np_bag, phrase_grams_dict, verbose=False):

    powerset_ixs = powerset(list(range(len(list_key_ix_phrases_dict))))
    vcs = []
    roots_dict_agg = {}
    print(list(powerset(list(range(len(list_key_ix_phrases_dict))))))
    n = max([z for _, _, z in np_bag])
    for ps_ix in powerset_ixs:
        cur_list_key_ix_phrases_dict = [list_key_ix_phrases_dict[ix] for ix in ps_ix]
        list_keys = [list(x.keys()) for x in cur_list_key_ix_phrases_dict]

        relevant_identifiers = [identifiers[ix] for ix in ps_ix]
        relevant_identifiers_str = '-'.join(relevant_identifiers)
        if verbose:
            print(list(product(*list_keys)))
        for key_seq in product(*list_keys):
            ixs_list = [set(item[k]) for k, item in zip(key_seq, cur_list_key_ix_phrases_dict)]
            ixs_tot = list(set.union(*ixs_list))
            key_seq_strd = ['_'.join(k) for k in key_seq]
            key_seqs_str = '-'.join(key_seq_strd)
            case_ident = '{0}.{1}#{2}'.format('cnt', key_seqs_str, relevant_identifiers_str)
            vc, roots_dict = get_vc_nps_from_phrases(np_bag, ixs_tot)
            vc = pd.DataFrame(vc, columns=[case_ident])
            vcs += [vc]
            for k, v in roots_dict.items():
                if k in roots_dict_agg.keys():
                    roots_dict_agg[k] |= roots_dict[k]
                else:
                    roots_dict_agg[k] = roots_dict[k]

    roots_dict_inv = {vv: k for k, v in roots_dict_agg.items() for vv in v}

    vcs = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='outer'), vcs)
    vcs['first_occ'] = vcs.index.to_series().apply(lambda x: np.min(phrase_grams_dict[roots_dict_inv[x]][x])/n)
    vcs = vcs.fillna(0)
    return vcs


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


# def collect_full_stats(carticle, keywords, nlp):
#     np_bag = None
#     np_dict = None
#     shifts = [0, 1]
#     shifted_vcs = []
#     shifted_roots = []
#     for shift in shifts:
#         vcs, roots, np_bag, np_dict = get_root0_stats(carticle, keywords, nlp, np_bag,
#                                                       np_dict, shift=shift)
#         shifted_vcs.append(vcs)
#         shifted_roots.append(roots)
#
#     tot_vcs = reduce(lambda x, y: x.merge(y[list(set(y.columns) - {'first_occ'})],
#                                           left_index=True, right_index=True, how='outer'),
#                      shifted_vcs)
#
#     for vc in shifted_vcs[1:]:
#         tot_vcs.update(vc['first_occ'])
#     tot_vcs = tot_vcs.fillna(0)
#
#     dd_agg = {}
#     for list_dicts in shifted_roots:
#         for plain_dict in list_dicts:
#             for k, v in plain_dict.items():
#                 if k in dd_agg.keys():
#                     dd_agg[k] |= set(v)
#                 else:
#                     dd_agg[k] = set(v)
#
#     vcs_second, roots, np_bag, np_dict = get_root0_stats(carticle, dd_agg, nlp, np_bag,
#                                                          np_dict, second_order=True, verbose=True)
#
#     dfr = pd.merge(vcs_second, tot_vcs[list(set(tot_vcs.columns) - {'first_occ'})],
#                    how='outer', left_index=True, right_index=True)
#     dfr.update(tot_vcs['first_occ'])
#     dfr = dfr.fillna(0)
#
#     return dfr
#
