import enchant
import numpy as np


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
    :param window:
    :parama prefix: True for prefix, False for suffix
    :parama verbose: verbosity
    :return: cleaned article
    """

    dir = 1 if prefix else -1
    article_out = [None] * len(article)
    last_page_a, last_page_b = article[-1], article[-2]

    if prefix:
        page_a_buff, page_b_buff = last_page_a[:window], last_page_b[:window]
    else:
        page_a_buff, page_b_buff = last_page_a[-window:], last_page_b[-window:]

    last_page_a_split, last_page_b_split = page_a_buff.split()[::dir], page_b_buff.split()[::dir]
    if verbose:
        print('len last_page_a_split {0} len last_page_b_split {1}'.format(len(last_page_a_split),
                                                                           len(last_page_b_split)))

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
        print('number of matching words: {0}'.format(n_matches))
        print('last page first word after prefix \"{0}\"'.format(last_page_a_split[n_matches]))

    ixs = list(range(len(article)))[::-1]
    for ix in ixs:
        page = article[ix]
        if prefix:
            page_buff = page[:window]
        else:
            page_buff = page[-window:]

        page_split = page_buff.split()[::dir]

        ind_a = sum(len(x) for x in page_split[:n_matches]) + n_matches

        if is_int(word_not_number) or any([w == word_not_number for w in page_split]):
            if prefix:
                article_out[ix] = page[ind_a:]
                if verbose:
                    print(page[:ind_a])
            else:
                article_out[ix] = page[:-ind_a]
                if verbose:
                    print(page[-ind_a:])

        else:
            article_out[ix] = page
    return article_out


# 1) check page breaks (no hyphen there), (beg, end).

def merge_page_breaks(article, window=150, verbose=False):
    article_out = [article[0]]
    d = enchant.Dict("en_US")

    for p2 in article[1:]:
        p1 = article_out.pop()
        ws1, ws2 = p1[-window:].split()[-1], p2[:window].split()[0]
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
