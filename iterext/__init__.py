import itertools

from . import recipes


__all__ = []

itertools_all = (
    'chain',
    'combinations',
    'combinations_with_replacement',
    'compress',
    'count',
    'cycle',
    'dropwhile',
    'groupby',
    'ifilter',
    'ifilterfalse',
    'imap',
    'islice',
    'izip',
    'izip_longest',
    'permutations',
    'product',
    'repeat',
    'starmap',
    'takewhile',
    'tee',
)


def update_globals(module, attrs):
    for attr in attrs:
        globals()[attr] = getattr(module, attr)

    __all__.extend(list(attrs))


update_globals(itertools, itertools_all)
update_globals(recipes, recipes.__all__)

del update_globals
