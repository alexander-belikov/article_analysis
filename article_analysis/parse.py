import enchant
import numpy as np
import re
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
from os.path import expanduser, join, isfile
from os import listdir
from .list_agg import ListAggregator
import gzip
import pickle
from copy import deepcopy


def is_int(x):
    try:
        int(x)
    except:
        return False
    return True


# 0) find common prefix in pages (odd, even) if there is a number, account for changes

def split_odds_evens(article):
    return article[::2], article[1::2]


def join_odds_evens(odds, evens):
    r = [None] * (len(odds) + len(evens))
    r[::2] = odds
    r[1::2] = evens
    return r


def eat_page_prefixes(article, window=150, verbose=False):
    """
    ocr specific parsing: article is scanned page by page,
    pages can be odd or even. Even and odd pages have different prefixes.
    :param article: list of pages (page is a string)
    :param window:
    :return: cleaned article
    """
    article_out = [None] * len(article)
    last_page_a, last_page_b = article[-1], article[-2]
    last_page_a_split, last_page_b_split = last_page_a[:window].split(), last_page_b[:window].split()
    cnt = 0

    while True:
        a, b = last_page_a_split[cnt], last_page_b_split[cnt]
        if a == b or (is_int(a) and is_int(b) and int(a) - int(b) == 2):
            cnt += 1
        else:
            break
    ind_a = sum(len(x) for x in last_page_a_split[:cnt]) + cnt
    if verbose:
        print('index of page beginning: {0}, last page first word after prefix \"{1}\"'.format(ind_a,
                                                                                               last_page_a_split[cnt]))
    ixs = list(range(len(article)))[::-1]
    for ix in ixs:
        page = article[ix]
        w = page[:window].split()[0]
        if w == last_page_a_split[0] or (is_int(w) and is_int(last_page_a_split[0])
                                         and (int(w) - int(last_page_a_split[0])) % 2 == 0):
            article_out[ix] = page[ind_a:]
            if verbose:
                print(page[:ind_a])
        else:
            article_out[ix] = page

    return article_out


def eat_page_suffixes(article, window=150, verbose=False):
    """
    ocr specific parsing: article is scanned page by page,
    pages can be odd or even. Even and odd pages have different prefixes.
    :param article: list of pages (page is a string)
    :param window:
    :parama verbose: verbosity
    :return: cleaned article
    """
    article_out = [None] * len(article)
    last_page_a, last_page_b = article[-1], article[-2]
    last_page_a_split, last_page_b_split = last_page_a[-window:].split()[::-1], last_page_b[-window:].split()[::-1]

    flags_eq = [True if a == b else False
                for a, b in zip(last_page_a_split, last_page_b_split)]
    flags_int = [True if (is_int(a) and is_int(b) and int(a) - int(b) == 2) else False
                 for a, b in zip(last_page_a_split, last_page_b_split)]

    flags = np.array(flags_eq) | np.array(flags_int)

    n_matches = 0
    while n_matches < len(flags) and flags[n_matches]:
        n_matches += 1

    is_number = [is_int(w) for w in last_page_a_split[:n_matches]]
    if not all(is_number):
        n_word = 0
        while n_word < len(is_number) and is_number[n_word]:
            n_word += 1
        word_not_number = last_page_a_split[n_word]
    else:
        word_not_number = -1

    if verbose:
        print('word_not_number: {0}'.format(word_not_number))

    if verbose:
        print('number of matching words: {0}, '
              'last page first word after prefix \"{1}\"'.format(n_matches, last_page_a_split[n_matches]))

    ixs = list(range(len(article)))[::-1]
    for ix in ixs:
        page = article[ix]
        page_split = page[-window:].split()[::-1]

        ind_a = sum(len(x) for x in page_split[:n_matches]) + n_matches

        if is_int(word_not_number) or any([w == word_not_number for w in page_split]):
            article_out[ix] = page[:-ind_a]
            if verbose:
                print(page[-ind_a:])
        else:
            article_out[ix] = page
    return article_out


def eat_page_prefix_suffix(article, prefix=True, window=150, verbose=False):
    """
    ocr specific parsing: article is scanned page by page,
    pages can be odd or even. Even and odd pages have different prefixes.
    :param article: list of pages (page is a string)
    :parama prefix: True for prefix, False for suffix
    :param window:
    :param verbose: verbosity
    :return: cleaned article
    """

    dir = 1 if prefix else -1
    article_out = [None] * len(article)

    last_page_a, last_page_b = article[-1], article[-2]
    window_a = min([len(last_page_a), window])
    window_b = min([len(last_page_b), window])

    if prefix:
        page_a_buff, page_b_buff = last_page_a[:window_a], last_page_b[:window_b]
    else:
        page_a_buff, page_b_buff = last_page_a[-window_a:], last_page_b[-window_b:]

    last_page_a_split, last_page_b_split = page_a_buff.split()[::dir], page_b_buff.split()[::dir]

    if len(last_page_a_split) > 0 and len(last_page_b_split) > 0:

        if verbose:
            print('window {0}'.format(window))
            print('len last_page_a_split {0} len last_page_b_split {1}'.format(len(last_page_a_split),
                                                                               len(last_page_b_split)))

        flags_eq = [True if a == b else False
                    for a, b in zip(last_page_a_split, last_page_b_split)]
        flags_int = [True if (is_int(a) and is_int(b) and int(a) - int(b) == 2) else False
                     for a, b in zip(last_page_a_split, last_page_b_split)]

        if verbose:
            print('len flag_eq : {0}, len flag_int {1}'.format(len(flags_eq), len(flags_int)))
        flags = np.array(flags_eq) | np.array(flags_int)
    else:
        flags = np.array([])
    n_matches = 0

    if flags.size != 0:
        while n_matches < len(flags) and flags[n_matches]:
            n_matches += 1

    is_number = [is_int(w) for w in last_page_a_split[:n_matches]]
    if is_number and not all(is_number):
        n_word = 0
        while n_word < len(is_number) and is_number[n_word]:
            n_word += 1
        word_not_number = last_page_a_split[n_word]
    else:
        word_not_number = -1

    if verbose:
        print('word_not_number: {0}'.format(word_not_number))

    if verbose:
        print('number of matching words: {0}'.format(n_matches))
        print('last page first word after prefix \"{0}\"'.format(last_page_a_split[n_matches]))

    ixs = list(range(len(article)))[::-1]
    for ix in ixs:
        page = article[ix]
        window_ = min([window, len(page)])
        if prefix:
            page_buff = page[:window_]
        else:
            page_buff = page[-window_:]

        page_split = page_buff.split()[::dir]

        ind_a = sum(len(x) for x in page_split[:n_matches]) + n_matches

        if is_int(word_not_number) or any([w == word_not_number for w in page_split]):
            if verbose:
                print('{0} {1}'.format(ind_a, len(page)))
            if prefix:
                article_out[ix] = page[ind_a:]
            else:
                if ind_a > 0:
                    article_out[ix] = page[:-ind_a]
                else:
                    article_out[ix] = page
        else:
            article_out[ix] = article[ix]
    return article_out


# 1) check page breaks (no hyphen there), (beg, end).

def merge_page_breaks(article, window=150, verbose=False):
    article_out = [article[0]]
    d = enchant.Dict("en_US")

    for p2 in article[1:]:
        p1 = article_out.pop()
        window1 = min([window, len(p1)])
        window2 = min([window, len(p2)])
        subphrase1, subphrase2 = p1[-window1:].split(), p2[:window2].split()
        if subphrase1 and subphrase2:
            ws1, ws2 = subphrase1[-1], subphrase2[0]
            w0 = ws1+ws2
            check_flag = d.check(w0)
            if d.suggest(w0):
                suggest_word = d.suggest(w0)[0]
            else:
                suggest_word = ''
            if len(suggest_word) == len(w0) and sum([c1 != c2 for c1, c2 in zip(suggest_word, w0)]):
                suggest_flag = True
            else:
                suggest_flag = False
            if check_flag or suggest_flag:
                if suggest_flag:
                    p1_work = p1[:-len(ws1)] + suggest_word
                else:
                    p1_work = p1[:-len(ws1)] + w0
                article_out.append(p1_work)
                p2_work = p2[len(ws2):]
                article_out.append(p2_work)
            else:
                article_out.append(p1)
                article_out.append(p2)
        else:
            article_out.append(p1)
            article_out.append(p2)

    # article_out[-1] = article[-1]
    return article_out


def check_page_breaks(article, window=150, nwords=3):

    for p in article[:]:
        ws1, ws2 = p[:window].split()[:nwords], p[-window:].split()[-nwords:]
        print(ws1, ws2)


# 2) check all "beg-" cases, if beg+end are in corpus


def merge_hyphens(phrase, checker_dict=None):
    print(phrase)
    new_phrase = [phrase[0]]
    if not checker_dict:
        checker_dict = enchant.Dict("en_US")
    if len(phrase) > 1:
        for w2 in phrase[1:]:
            w1 = new_phrase.pop()
            print(w1, w2)
            if w1[-1] == '-' and checker_dict.check(w1[:-1] + w2):
                new_phrase.append(w1[:-1] + w2)
            else:
                new_phrase.append(w1)
                new_phrase.append(w2)
        return new_phrase
    else:
        return phrase


def merge_hyphenated_words(super_phrase, checker_dict=None):
    #TODO simpligy if else logic
    if not checker_dict:
        checker_dict = enchant.Dict("en_US")

    super_phrase_hyphed = super_phrase[:1]
    super_phrase_copy = list(super_phrase[1:])

    while super_phrase_copy:
        w1 = super_phrase_copy.pop(0)
        if w1 == '-':
            if super_phrase_copy:
                w2 = super_phrase_copy.pop(0)
                if checker_dict.check(super_phrase_hyphed[-1] + w2) and w2.isalpha():
                    w0 = super_phrase_hyphed.pop()
                    super_phrase_hyphed.append(w0 + w2)
                else:
                    super_phrase_hyphed.append(w1)
            else:
                super_phrase_hyphed.append(w1)
        else:
            super_phrase_hyphed.append(w1)
    return super_phrase_hyphed


def split_into_phrases(tokens_list):
    phrases = []
    cur_phrase = []
    for token1, token2 in zip(tokens_list[:-1], tokens_list[1:]):
        cur_phrase.append(token1)
        if token1 == '.' and token2[0].isupper():
            phrases.append(cur_phrase)
            cur_phrase = []
    cur_phrase.append(tokens_list[-1])
    phrases.append(cur_phrase)
    return phrases


def eat_stopwords(phrase, stop_tokens):
    return [w for w in phrase if w not in stop_tokens]


def transform_article(article, verbose=False):

    if verbose:
        print('number of pages: {0}'.format(len(article)))

    article = [page for page in article if len(page) > 5]
    odd_pages, even_pages = split_odds_evens(article)

    # eat prefix/suffix of odd pages
    if len(odd_pages) > 1:
        odds2 = eat_page_prefix_suffix(odd_pages, prefix=True, verbose=verbose)
        odds3 = eat_page_prefix_suffix(odds2, prefix=False, verbose=verbose)
    else:
        odds3 = odd_pages

    # eat prefix/suffix of even pages
    if len(even_pages) > 1:
        evens2 = eat_page_prefix_suffix(even_pages, prefix=True, verbose=verbose)
        evens3 = eat_page_prefix_suffix(evens2, prefix=False, verbose=verbose)
    else:
        evens3 = even_pages

    # join odd and even pages
    article2 = join_odds_evens(odds3, evens3)

    if verbose:
        print('page lens are {0}'.format([len(x) for x in article2]))

    # merge page breaks
    article3 = merge_page_breaks(article2, verbose=True)

    # aggregate pages into a string
    sagg = ''
    for s in article3:
        sagg += s

    # split only if a white space follows and the previous letter is capital
    tokenized_agg = re.findall(r"[\w']+|[.,!?;:-]", sagg)
    super_phrase_hyphed = merge_hyphenated_words(tokenized_agg)
    phrases = split_into_phrases(super_phrase_hyphed)
    return phrases


def lower_rm_stopwords_digits(phrases, eat_numbers=True, lower_case=True, rm_stopwords=True):
    phrases = deepcopy(phrases)
    if lower_case:
        phrases = [[w.lower() for w in phrase] for phrase in phrases]

    swords = set(stopwords.words('english')) | set(list('.,!?;:-'))

    if rm_stopwords:
        phrases = [eat_stopwords(phrase, swords) for phrase in phrases]
    if eat_numbers:
        phrases = [[w for w in phrase if not w.isdigit()] for phrase in phrases]
    phrases = [phrase for phrase in phrases if phrase]
    return phrases


def compute_ngrams_positions(article, lowest_order=1, highest_order=5):
    # aggregate ngrams for an article
    ngrams_dict = {k: Counter() for k in range(lowest_order, highest_order+1)}
    for order in range(lowest_order, highest_order+1):
        cnt = Counter()
        for phrase in article:
            cnt += Counter(ngrams(phrase, order))
        ngrams_dict[order] += cnt
    return ngrams_dict

# def compute_ngrams(article, lowest_order=1, highest_order=5):
#     # aggregate ngrams for an article
#     ngrams_dict = {k: Counter() for k in range(lowest_order, highest_order+1)}
#     for order in range(lowest_order, highest_order+1):
#         cnt = Counter()
#         for phrase in article:
#             if order > 1:
#                 cnt += Counter(ngrams(phrase, order))
#             else:
#                 cnt += Counter((x[0] for x in ngrams(phrase, order)))
#         ngrams_dict[order] += cnt
#     return ngrams_dict


def compute_ngrams(article, lowest_order=1, highest_order=5, counter_type='counter'):
    # aggregate ngrams for an article
    if counter_type == 'list_agg':
        cnt_class = ListAggregator
    else:
        cnt_class = Counter
    ngrams_dict = {k: cnt_class() for k in range(lowest_order, highest_order+1)}
    for order in range(lowest_order, highest_order+1):
        cnt = cnt_class()
        for phrase, ii in zip(article, range(len(article))):
            if order > 1:
                cc2 = Counter(ngrams(phrase, order))
                if counter_type == 'list_agg':
                    cnt.from_counter(cc2, ii)
                else:
                    cnt += cc2
            else:
                cc2 = Counter((x[0] for x in ngrams(phrase, order)))
                if counter_type == 'list_agg':
                    cnt.from_counter(cc2, ii)
                else:
                    cnt += cc2
        ngrams_dict[order] += cnt
    return ngrams_dict


def get_present_keys(fpath, prefix, suffix):
    suffix_len = len(suffix)
    prefix_len = len(prefix)
    files = [f for f in listdir(fpath) if isfile(join(fpath, f)) and
             (f[-suffix_len:] == suffix and f[:prefix_len] == prefix)]
    ints = [int(f.split('_')[-1].split('.')[0]) for f in files]
    if ints:
        max_index = max(ints)
    else:
        max_index = 0
    present_keys = set()
    for f in files:
        with gzip.open(join(fpath, f)) as fp:
            item = pickle.load(fp)
            present_keys |= set(list(item.keys()))
    return present_keys, max_index


def split_corpus(corpus, chunk_size=200):
    corpus_keys = list(corpus.keys())
    keys_chunks = [corpus_keys[k:k+chunk_size] for k in range(0, len(corpus_keys), chunk_size)]
    corpus_chunks = [{k: corpus[k] for k in chunk} for chunk in keys_chunks]
    return corpus_chunks


def transform_counter(cnt, how='lemma', lmtzr=None, counter_type='counter'):
    if counter_type == 'list_agg':
        cnt_class = ListAggregator
    else:
        cnt_class = Counter

    if how == 'lemma':
        if not lmtzr:
            lmtzr = WordNetLemmatizer()
        cnt_new = cnt_class()
        for item, cnt in cnt.items():
            if isinstance(item, str):
                k = lmtzr.lemmatize(item)
            else:
                k = tuple([lmtzr.lemmatize(x) for x in item])
            cnt_new += cnt_class({k: cnt})
        return cnt_new
    else:
        return cnt


def transform_chunk(chunk, lmtzr=None):
    if not lmtzr:
        lmtzr = WordNetLemmatizer()
    new_chunk = {}
    keys = sorted(chunk.keys())
    for k in keys[:]:
        orders = sorted(chunk[k].keys())
        new_chunk[k] = {}
        for j in orders[:]:
            new_chunk[k][j] = transform_counter(chunk[k][j], how='lemma', lmtzr=lmtzr)
    return new_chunk


def get_indices(fpath, prefix, suffix):
    suffix_len = len(suffix)
    prefix_len = len(prefix)
    files = [f for f in listdir(fpath) if isfile(join(fpath, f)) and
             (f[-suffix_len:] == suffix and f[:prefix_len] == prefix)]
    ints = [int(f.split('_')[-1].split('.')[0]) for f in files]
    return sorted(ints)


def get_chunk(fpath, prefix, index):
    fname = join(fpath, '{0}_{1}.pgz'.format(prefix, index))
    with gzip.open(fname) as fp:
        item = pickle.load(fp)
        return item
    return None
