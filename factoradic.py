
from bisect import bisect_left

def factoradic_digits(n, size):
    digits = []
    base = 1
    while len(digits) < size:
        digits.append(n % base)
        n //= base
        base += 1
    return digits

def factoradic(digits):
    n = 0
    prod = 1
    base = 1
    for digit in digits:
        if not 0 <= digit < base:
            raise ValueError(f'digit {digit} out of range for base {base}')
        n += digit * prod
        prod *= base
        base += 1
    return n

def permute(items, n):
    #     |abcd   0102
    #    c|abd    010
    #   ca|bd     01
    #  cad|b      0
    # cadb|
    perm = []
    for i in reversed(factoradic_digits(n, len(items))):
        perm.append(items.pop(i))
    return perm

def unpermute(perm):
    # cadb|
    #  cad|b  0
    #   ca|bd  01
    #    c|abd  010
    #     |abcd  0102
    items = []
    digits = []
    for item in reversed(perm):
        i = bisect_left(items, item)
        items.insert(i, item)
        digits.append(i)
    return items, factoradic(digits)

if __name__ == '__main__':
    from itertools import permutations

    def test():
        items = [1, 2, 3, 4]
        for i, perm in enumerate(sorted(permutations(items))):
            assert permute(items.copy(), i) == list(perm), f'permute({items}, {i}) == {perm}'
            assert unpermute(perm) == (items, i), f'unpermute({perm}) == ({items}, {i})'

    test()
