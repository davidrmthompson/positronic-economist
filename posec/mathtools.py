import math


def isDistribution(d):
    ''' d is a list of probabilities '''
    if min(d) < 0.0:
        return False
    if max(d) > 1.0:
        return False
    if abs(sum(d) - 1.0) > 0.000000000001:
        return False
    return True


def isDelta(d):
    if not isDistribution(d):
        print d
    assert isDistribution(d)
    return max(d) == sum(d)


def cartesianProduct(setOfSets, prefixes=[[]]):
    if not setOfSets:
        return prefixes
    output = []
    for prefix in prefixes:
        for element in setOfSets[0]:
            output.append(prefix + [element])
    return cartesianProduct(setOfSets[1:], output)
# Recursion is cute, but iterators are better.


def cartesianProduct(listOfLists):
    n = 1L
    lens = map(len, listOfLists)
    for s in listOfLists:
        n *= len(s)
    d = len(listOfLists)
    for i in xrange(n):
        element = []
        for j in range(d):
            i, index = divmod(i, lens[j])
            element.append(listOfLists[j][index])
        yield tuple(element)


def permutations(S):
    S = list(S)
    if len(S) == 1:
        return [S]
    output = []
    for i in range(len(S)):
        for perm in permutations(S[:i] + S[i + 1:]):
            output.append([S[i]] + perm)
    return output


def subsetPermutations(S):
    output = []
    for S2 in powerSet(S):
        if not S2:
            output.append([])
        else:
            output += permutations(list(S2))
    return output


def intToBitVector(n, minBits=0):
    output = []
    while n > 0:
        bit = n % 2
        output = [bit] + output
        n /= 2
    while len(output) < minBits:
        output = [0] + output
    return output


def powerSet(S):
    n = len(S)
    output = []
    for i in range(int(math.pow(2, n))):
        bv = intToBitVector(i, n)
        subset = set([S[j] for j in range(n) if bv[j] == 1])
        output.append(subset)
    return output
