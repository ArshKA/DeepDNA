import itertools

bases = ['A', 'T', 'G', 'C']

single_map = {i: kmer for i, kmer in enumerate(bases)}

def get_kmer_map(k, inverse=False):
    # starting = [' ' + ''.join(p) for p in itertools.product(bases, repeat=2)]
    middle = [''.join(p) for p in itertools.product(bases, repeat=k)]
    # ending = [''.join(p) + ' ' for p in itertools.product(bases, repeat=2)]
    if inverse:
        kmer_map = {i: kmer for i, kmer in enumerate(middle)}
    else:
        kmer_map = {kmer: i for i, kmer in enumerate(middle)}
    return kmer_map

