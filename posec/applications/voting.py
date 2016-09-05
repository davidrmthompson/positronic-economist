import posec
from posec import mathtools
#import mathtools
import sys
import traceback

# Utilities for calculating the dominated strategies


def leastFavorites(setting, i, theta_i):
    ''' returns a list of i's least favorite outcomes (or an empty list if the computation can't be done, due to CTD) '''
    try:
        O = setting.O
        minU = min([setting.u(i, {i: theta_i}, o, None) for o in O])
        return [o for o in O if setting.u(i, {i: theta_i}, o, None) == minU]
    except:
        print "Exception", i, theta_i, sys.exc_info()
        traceback.print_tb(sys.exc_info()[-1])
        return []


def mostFavorites(setting, i, theta_i):
    ''' returns a list of i's most favorite outcomes (or an empty list if the computation can't be done due to CTD) '''
    try:
        O = setting.O
        maxU = max([setting.u(i, {i: theta_i}, o, None) for o in O])
        return [o for o in O if setting.u(i, {i: theta_i}, o, None) == maxU]
    except:
        print "Exception", i, theta_i, sys.exc_info()
        traceback.print_tb(sys.exc_info()[-1])
        return []


def isRanking(setting, i, theta_i):
    ''' Returns outcomes listed from i's most to least favorite '''
    try:
        scores = [(setting.u(i, {i: theta_i}, o, None), o) for o in setting.O]
        scores.sort()
        scores.reverse()
        return [o for s, o in scores]
    except:
        print "Exception", i, theta_i, sys.exc_info()
        traceback.print_tb(sys.exc_info()[-1])
        return []


class AbstractVotingMechanism(posec.Mechanism):

    ''' Provides a bunch of support features for real voting mechanisms '''
    randomTieBreak = True  # Otherwise, choose lexigraphic first
    allowAbstain = False
    removeDominatedStrategies = False

    def __init__(self, randomTieBreak=True, removeDominatedStrategies=False, allowAbstain=False):
        self.randomTieBreak = randomTieBreak
        self.removeDominatedStrategies = removeDominatedStrategies
        self.allowAbstain = allowAbstain

    def A(self, setting, i, theta_i):
        ''' returns self.actions(setting, i, theta_i) after (optionally) removing dominated strategies and adding an abstain action

Allows RDS, etc to be re-used '''
        A = list(self.actions(setting, i, theta_i))
        if self.removeDominatedStrategies:
            M = mostFavorites(setting, i, theta_i)
            # Nothing is dominated if the agent doesn't care (or has prefs that
            # aren't so easily determined)
            if set(M) != set(setting.O) and len(M) != 0:
                A = filter(
                    lambda x: not self.dominated(setting, i, theta_i, x), A)

        for a in A:
            a.truthful = self.truthful(setting, i, theta_i, a)

        if self.allowAbstain:
            A.append(None)
        return tuple(A)

    def outcome(self, scores):
        ''' Subroutine for M, identifies the maximal-score candidates and breaks ties appropriately '''
        maxScore = max(scores.values())
        winners = [o for o in scores.keys() if scores[o] == maxScore]
        if self.randomTieBreak:
            return posec.UniformDistribution(winners)
        return min(winners)


class VoteTuple(tuple):

    ''' Like a normal tuple, but it can also mark an action as truthful '''
    truthful = False

    def __repr__(self):
        return str((tuple(list(self)), self.truthful))


class Plurality(AbstractVotingMechanism):

    def actions(self, setting, i, theta_i):
        A = [VoteTuple((o,)) for o in setting.O]
        return A

    def truthful(self, setting, i, theta_i, a):
        return a[0] in mostFavorites(setting, i, theta_i)

    def dominated(self, setting, i, theta_i, a):
        L = leastFavorites(setting, i, theta_i)
        return a[0] in L

    def M(self, setting, a_N):
        scores = dict([(o, a_N.count((o,))) for o in setting.O])
        return self.outcome(scores)


class Approval(AbstractVotingMechanism):

    def actions(self, setting, i, theta_i):
        return [VoteTuple(S) for S in mathtools.powerSet(setting.O) if len(S) > 0]

    def truthful(self, setting, i, theta_i, a):
        return set(a) == set(mostFavorites(setting, i, theta_i))
        # FIXME: Is this actually what I mean?

    def dominated(self, setting, i, theta_i, a):
        if not all([o in a for o in mostFavorites(setting, i, theta_i)]):
            return True
        if not all([o not in a for o in leastFavorites(setting, i, theta_i)]):
            return True
        return False

    def M(self, setting, a_N):
        scores = {}
        for o in setting.O:
            scores[o] = a_N.sum(
                [a for a in a_N.actions if a != None and o in a])
        return self.outcome(scores)


class kApproval(Approval):

    def __init__(self, k, randomTieBreak=True, removeDominatedStrategies=False, allowAbstain=False):
        self.k = k
        self.randomTieBreak = randomTieBreak
        self.removeDominatedStrategies = removeDominatedStrategies
        self.allowAbstain = allowAbstain

    def actions(self, setting, i, theta_i):
        A = [s for s in mathtools.powerSet(setting.O) if len(s) == self.k]
        return map(VoteTuple, A)

    def truthful(self, setting, i, theta_i, a):
        r = isRanking(setting, i, theta_i)
        # print "Ranking:",i, theta_i, r
        if not r:
            return False
        return set(a) == set(r[:self.k])

    def dominated(self, setting, i, theta_i, a):
        M = mostFavorites(setting, i, theta_i)
        L = leastFavorites(setting, i, theta_i)
        if any([(o in a) for o in L]):
            return not all([(o in a) for o in M])
        return False


class Veto(Approval):

    def actions(self, setting, i, theta_i):
        A = [s for s in mathtools.powerSet(
            setting.O) if len(s) == len(setting.O) - 1]
        return map(VoteTuple, A)

    def truthful(self, setting, i, theta_i, a):
        return False

    def dominated(self, setting, i, theta_i, a):
        M = mostFavorites(setting, i, theta_i)
        if not all([o in a for o in M]):
            return True
        return False


class Borda(AbstractVotingMechanism):

    def actions(self, setting, i, theta_i):
        A = mathtools.permutations(setting.O)
        return map(VoteTuple, A)

    def truthful(self, setting, i, theta_i, a):
        return False

    def dominated(self, setting, i, theta_i, a):
        M = mostFavorites(setting, i, theta_i)
        L = leastFavorites(setting, i, theta_i)
        for m in M:
            for l in L:
                if a.index(m) < a.index(l):
                    return True
        return False

    def M(self, setting, a_N):
        scores = {}
        for o in setting.O:
            # print type(a_N.actions)
            weights = [a.index(o) for a in a_N.actions if a != None]
            scores[o] = a_N.weightedSum(a_N.actions, weights)
        return self.outcome(scores)


class InstantRunoff(AbstractVotingMechanism):

    def __init__(self, allowAbstain=False):
        ''' FIXME: So far, ties must be broken alphabetically '''
        self.randomTieBreak = False
        self.removeDominatedStrategies = False
        self.allowAbstain = allowAbstain

    def actions(self, setting, i, theta_i):
        A = mathtools.permutations(setting.O)
        return map(VoteTuple, A)

    def truthful(self, setting, i, theta_i, a):
        return False

    def dominated(self, setting, i, theta_i, a):
        ''' Dominance checking is not implemented by InstantRunoff '''
        raise Exception(
            "Dominance checking is not implemented by InstantRunoff")

    def M(self, setting, a_N):
        winners = [o for o in setting.O]
        while len(winners) != 1:
            scores = []
            for o in winners:
                actionsForO = []
                for a in a_N.actions:
                    if all([a.index(o2) <= a.index(o) for o2 in winners]):
                        actionsForO.append(a)
                # print winners,o,actionsForO
                scores.append((a_N.sum(actionsForO), o))
            scores.sort()
            winners = [o for s, o in scores[1:]]  # cut off the lowest score
        return winners[0]

MECHANISM_CLASSES = [
    Plurality, Approval, kApproval, Veto, Borda, InstantRunoff]


def _sortedTupleSet(S):
    S2 = []
    for e in S:
        if e not in S2:
            S2.append(e)
    S2.sort()
    return tuple(S2)


def settingFromRankings(rankings, truthfulness=True):
    n = len(rankings)
    O = _sortedTupleSet(rankings[0])
    Theta = map(tuple, rankings)
    AbstainEps = 0.000001
    TruthfulEps = 0.000001 * truthfulness

    def u(i, theta, o, a_i):
        truth = False
        if a_i != None:
            truth = a_i.truthful
        return len(O) - theta[i].index(o) + AbstainEps * (a_i == None) - 1 + TruthfulEps * truth
    return posec.Setting(n, O, Theta, u)


def urn_model_setting(n, m, a, seed):
    outcomes = [(chr(i + 65)) for i in range(m)]
    possible_rankings = mathtools.permutations(outcomes)
    import random as r
    r.seed(seed)
    rankings = []
    for i in range(n):
        ranking = r.choice(possible_rankings)
        rankings.append(ranking)
        for _ in range(a):
            possible_rankings.append(ranking)
    return settingFromRankings(rankings)


def uniform_Setting(n, m, seed):
    return urn_model_setting(n, m, 0, seed)


def impartialAnonymousCulture_Setting(n, m, seed):
    return urn_model_setting(n, m, 1, seed)


def threeUrn_Setting(n, m, seed):
    return urn_model_setting(n, m, 3, seed)


def jMajority_Setting(n, m, j, seed):
    outcomes = [(chr(i + 65)) for i in range(m)]
    possible_rankings = mathtools.permutations(outcomes)
    k = len(possible_rankings) - 3
    import random as r
    r.seed(seed)
    majorities = []
    while True:
        maj = r.choice(possible_rankings)
        if maj in majorities:
            continue
        majorities.append(maj)
        if len(majorities) == j:
            break
    for i in range(k):
        possible_rankings += majorities
    rankings = [r.choice(possible_rankings) for i in range(n)]
    return settingFromRankings(rankings)


def twoMajority_Setting(n, m, seed):
    return jMajority_Setting(n, m, 2, seed)


def threeMajority_Setting(n, m, seed):
    return jMajority_Setting(n, m, 3, seed)


def isSinglePeaked(ranking):
    top = ranking[0]
    n = len(ranking)
    for j in range(1, n - 1):
        for k in range(j + 1, n):
            rj = ranking[j]
            rk = ranking[k]
            if rj > top:
                if rk > top and rk < rj:
                    return False
            if rj < top:
                if rk < top and rk > rj:
                    return False
    return True


def singlePeaked_Setting(n, m, seed):
    import random as r
    r.seed(seed)
    outcomes = [(chr(i + 65)) for i in range(m)]
    possible_rankings = filter(
        isSinglePeaked, mathtools.permutations(outcomes))
    # print possible_rankings
    rankings = [r.choice(possible_rankings) for i in range(n)]
    return settingFromRankings(rankings)


def uniformXY_Setting(n, m, seed):
    import random as r
    r.seed(seed)

    def toText(point):
        return "(%.2f,%.2f)" % (point)
    outcomes = [(r.random(), r.random()) for i in range(m)]

    def distanceSquared(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
    rankings = []
    for i in range(n):
        position = r.random(), r.random()
        distOutcomeList = [(distanceSquared(position, o), o) for o in outcomes]
        distOutcomeList.sort()
        ranking = [toText(o) for d, o in distOutcomeList]
        print position, ranking
        rankings.append(ranking)
    return settingFromRankings(rankings)
