""" Itertools recipes from
        http://docs.python.org/library/itertools.html#recipes
"""

import collections
import functools
from itertools import chain, combinations, count, cycle, groupby, imap
from itertools import ifilterfalse, islice, izip, izip_longest, repeat, starmap
from itertools import takewhile, tee
import operator
import random


__all__ = (
    'chunks',
    'consume',
    'dotproduct',
    'flatten',
    'grouper',
    'iter_except',
    'ncycles',
    'nth',
    'padnone',
    'pairwise',
    'powerset',
    'quantify',
    'random_combination',
    'random_combination_with_replacement',
    'random_permutation',
    'random_product',
    'repeatfunc',
    'roundrobin',
    'tabulate',
    'take',
    'tee_lookahead',
    'unique_everseen',
    'unique_justseen'
)


def flatten(listOfLists):
    """Flatten one level of nesting

    Examples:
        >>> list(flatten([]))
        []

        >>> list(flatten([[], [], []]))
        []

        >>> list(flatten([[1, 2, 3]]))
        [1, 2, 3]

        >>> list(flatten([[1, 2, 3], [11, 22, 33]]))
        [1, 2, 3, 11, 22, 33]
    """
    return chain.from_iterable(listOfLists)


def chunks(iterable, size=20):
    """Generator that iterates over an iterable in size chunks

    >>> list(chunks('ABCDEFG', size=3))
    [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)]

    >>> list(chunks('ABCDEFG', size=3))
    [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)]
    """
    stop = object()

    for chunk in grouper(size, iterable, fillvalue=stop):
        if chunk[-1] is stop:
            is_not_stop = functools.partial(operator.is_not, stop)

            yield tuple(takewhile(is_not_stop, chunk))

            break

        yield chunk


def grouper(n, iterable, fillvalue=None):
    """Return groups of length `n`.

    Examples with and without a fill character:

        >>> list(grouper(3, 'ABCDEFG', 'x'))
        [('A', 'B', 'C'), ('D', 'E', 'F'), ('G', 'x', 'x')]

        >>> [''.join(s) for s in grouper(3, 'ABCDEFG', 'x')]
        ['ABC', 'DEF', 'Gxx']

        >>> [''.join(s) for s in grouper(3, 'ABCDEFG', '')]
        ['ABC', 'DEF', 'G']
    """
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


def nth(iterable, n, default=None):
    """Return the nth item or a default value

    Examples:
        >>> nth([], 0) is None
        True

        >>> values = [10, 20, 30]
        >>> [nth(values, 0), nth(values, 1), nth(values, 2), nth(values, 3)]
        [10, 20, 30, None]
    """
    return next(islice(iterable, n, None), default)


def pairwise(iterable):
    """
    Examples,
        >>> list(pairwise([1,2,3]))
        [(1, 2), (2, 3)]

        >>> list(pairwise([1]))
        []

        >>> list(pairwise([]))
        []
    """
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


def take(n, iterable):
    """Return first n items of the iterable as a list.

    Example::
        >>> take(3, 'ABCDEFG')
        ['A', 'B', 'C']

        >>> take(3, '')
        []
    """
    return list(islice(iterable, n))


def tabulate(function, start=0):
    """Return function(0), function(1), ...

    Example::
        >>> t = tabulate(str)
        >>> [next(t), next(t), next(t)]
        ['0', '1', '2']

        >>> t = tabulate(str, 5)
        >>> [next(t), next(t), next(t)]
        ['5', '6', '7']
    """
    return imap(function, count(start))


def consume(iterator, n):
    """Advance the iterator n-steps ahead. If n is none, consume entirely.

    Example::
        >>> it = iter([1, 2, 3, 4, 5])
        >>> consume(it, None)
        >>> list(it)
        []

        >>> it = iter([1, 2, 3, 4, 5])
        >>> consume(it, 3)
        >>> list(it)
        [4, 5]
    """
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


def quantify(iterable, pred=bool):
    """Count how many times the predicate is true.

    Example::
        >>> quantify([0, 1, 1, 0, 0])
        2

        >>> quantify([0, 1, 1, 0, 0])
        2

        >>> quantify([1, 'a', 2, 'b', '3'], operator.isNumberType)
        2
    """
    return sum(imap(pred, iterable))


def padnone(iterable):
    """Return the sequence elements and then return None indefinitely.

    Useful for emulating the behavior of the built-in map() function.

    Examples::
        >>> take(7, padnone([1, 2, 3]))
        [1, 2, 3, None, None, None, None]
    """
    return chain(iterable, repeat(None))


def ncycles(iterable, n):
    """Return the sequence elements n times

    Example::
        >>> list(ncycles([1, 2], 3))
        [1, 2, 1, 2, 1, 2]
    """
    return chain.from_iterable(repeat(tuple(iterable), n))


def dotproduct(vec1, vec2):
    """Return the dot product of `vec1` and `vec2`

    Example::
        >>> dotproduct((10, 20, 30), (2, 0.25, 0.5))
        40.0
    """
    return sum(imap(operator.mul, vec1, vec2))


def repeatfunc(func, times=None, *args):
    """Repeat calls to func with specified arguments.

    Example::
        >>> import operator
        >>> list(repeatfunc(operator.mul, 3, 2, 6))
        [12, 12, 12]

        >>> list(repeatfunc(','.join, 3, 'abc'))
        ['a,b,c', 'a,b,c', 'a,b,c']

        >>> l = []
        >>> consume(repeatfunc(l.append, None, 'a'), 3)
        >>> l
        ['a', 'a', 'a']
    """
    if times is None:
        return starmap(func, repeat(args))
    return starmap(func, repeat(args, times))


def roundrobin(*iterables):
    """roundrobin('ABC', 'D', 'EF') --> A D E B F C

    Example::
        >>> ''.join(roundrobin('ABC', 'D', 'EF'))
        'ADEBFC'
    """
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))


def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    Example::
        >>> list(powerset([1, 2, 3]))
        [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def unique_everseen(iterable, key=None):
    """List unique elements, preserving order. Remember all elements ever seen.

    Example::
        >>> ''.join(unique_everseen('AAAABBBCCDAABBB'))
        'ABCD'

        >>> ''.join(unique_everseen('ABBCcAD'))
        'ABCcD'

        >>> ''.join(unique_everseen('ABBCcAD', str.lower))
        'ABCD'
    """
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in ifilterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def unique_justseen(iterable, key=None):
    """List unique elements, preserving order. Remember only the element just
    seen.

    Example::
        >>> ''.join(unique_justseen('AAAABBBCCDAABBB'))
        'ABCDAB'

        >>> ''.join(unique_justseen('ABBCcAD'))
        'ABCcAD'

        >>> ''.join(unique_justseen('ABBCcAD', str.lower))
        'ABCAD'
    """
    # unique_justseen('AAAABBBCCDAABBB') --> A B C D A B
    # unique_justseen('ABBCcAD', str.lower) --> A B C A D
    return imap(next, imap(operator.itemgetter(1), groupby(iterable, key)))


def iter_except(func, exception, first=None):
    """ Call a function repeatedly until an exception is raised.

    Converts a call-until-exception interface to an iterator interface.
    Like __builtin__.iter(func, sentinel) but uses an exception instead
    of a sentinel to end the loop.

    Example::
        >>> list(iter_except([1, 2, 3].pop, IndexError))
        [3, 2, 1]

    More examples:
        bsddbiter = iter_except(db.next, bsddb.error, db.first)
        heapiter = iter_except(functools.partial(heappop, h), IndexError)
        dictiter = iter_except(d.popitem, KeyError)
        dequeiter = iter_except(d.popleft, IndexError)
        queueiter = iter_except(q.get_nowait, Queue.Empty)
        setiter = iter_except(s.pop, KeyError)

    """
    try:
        if first is not None:
            yield first()
        while 1:
            yield func()
    except exception:
        pass


def random_product(*args, **kwds):
    """Random selection from itertools.product(*args, **kwds)

    Examples,

        >>> import random; random.seed(1)

        >>> random_product('123', 'abc')
        ('1', 'c')

        >>> random_product('123', 'abc', repeat=2)
        ('3', 'a', '2', 'b')
    """
    pools = map(tuple, args) * kwds.get('repeat', 1)
    return tuple(random.choice(pool) for pool in pools)


def random_permutation(iterable, r=None):
    """Random selection from itertools.permutations(iterable, r)

    Examples,

        >>> import random; random.seed(1)

        >>> random_permutation('123abc')
        ('1', 'b', 'a', 'c', '3', '2')

        >>> random_permutation('123abc')
        ('a', 'c', '1', 'b', '2', '3')

        >>> random_permutation('123abc', r=3)
        ('b', '1', '2')

        >>> random_permutation('123abc', r=3)
        ('b', '2', 'a')
    """
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))


def random_combination(iterable, r):
    """Random selection from itertools.combinations(iterable, r)

    Examples,

        >>> import random; random.seed(1)

        >>> random_combination('123abc', 3)
        ('1', 'a', 'b')

        >>> random_combination('123abc', 3)
        ('2', '3', 'c')
    """
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(xrange(n), r))
    return tuple(pool[i] for i in indices)


def random_combination_with_replacement(iterable, r):
    """Random selection from itertools.combinations_with_replacement(
    iterable, r)

    Examples,

        >>> import random; random.seed(1)

        >>> random_combination_with_replacement('123abc', 3)
        ('1', 'b', 'c')

        >>> random_combination_with_replacement('123abc', 3)
        ('2', '3', '3')
    """
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.randrange(n) for i in xrange(r))
    return tuple(pool[i] for i in indices)


def tee_lookahead(t, i):
    """Inspect the i-th upcoming value from a tee object while leaving the tee
    object at its current position.

    Raise an IndexError if the underlying iterator doesn't
    have enough values.

    Examples,

        >>> from itertools import tee

        >>> t = tee('abc')

        >>> tee_lookahead(t[0], 1), tee_lookahead(t[1], 1)
        ('b', 'b')

        >>> tee_lookahead(t[0], 0), tee_lookahead(t[1], 0)
        ('a', 'a')

        >>> tee_lookahead(t[0], 1), tee_lookahead(t[1], 4)
        Traceback (most recent call last):
            ...
        IndexError: 4

        >>> tee_lookahead(t[0], 2), tee_lookahead(t[1], 2)
        ('c', 'c')

    And the tee objects in t are still at their respective start positions:

        >>> zip(*t)
        [('a', 'a'), ('b', 'b'), ('c', 'c')]
    """
    for value in islice(t.__copy__(), i, None):
        return value
    raise IndexError(i)
