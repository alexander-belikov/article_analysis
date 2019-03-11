import spacy
import networkx as nx


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


def get_np_candidates(iphrases, article, nlp, window=1):
    """

    :param iphrases: indices of phrases of interest in the articel
    :param article: list of phrases
    :param nlp: spacy nlp module
    :window window:
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
            supp_chunk = [c for c in chunk if not c.is_stop and
                          not c.text.lower() in nlp.Defaults.stop_words and
                          not c.like_num]
            if supp_chunk and chunk.root.pos_ == 'NOUN':
                edge_slist = [[(c.lemma_, d.lemma_) for d in c.children if d in supp_chunk] for c in supp_chunk]
                edge_list = [e for sublist in edge_slist for e in sublist]
                g = nx.DiGraph()
                edge_list += [('#', chunk.root.lemma_)]
                g.add_edges_from(edge_list)
                supp_chunk2 = [c.lemma_ for c in supp_chunk]
                chunks.append((chunk.root.lemma_, tuple(supp_chunk2), g))
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

