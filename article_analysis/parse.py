
def is_int(x):
    try:
        int(x)
    except:
        return False
    return True


def split_odds_evens(article):
    return article[::2], article[1::2]


def join_odds_evens(odds, evens):
    r = [None] * (len(odds) + len(evens))
    r[::2] = odds
    r[1::2] = evens
    return r


def eat_page_prefixes(article, window=150):
    article_out = [[]] * len(article)
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
    ixs = list(range(len(article)))[::-1]
    for ix in ixs:
        page = article[ix]
        if page[:window].split()[0] == last_page_a_split[0]:
            article_out[ix] = page[ind_a:]
        else:
            article_out[ix] = page

    return article_out
