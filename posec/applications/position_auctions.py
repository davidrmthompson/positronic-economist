''' This is strictly for no-externality position auctions. '''
from collections import namedtuple
import posec
import string
import random as r
import math


class Permutations:

    def __init__(self, members, lengths=None):
        self.members = members
        self.lengths = lengths

    def __contains__(self, element):
        try:
            if self.lengths is not None:
                if len(element) not in self.lengths:  # Unallowed length
                    return False
            if len(element) != len(set(element)):  # Duplicate entries
                return False
            return all([(c in self.members) for c in element])
        except:  # Length or iteration has failed
            return False

    def __eq__(self, other):
        if not isinstance(other, Permutations):
            return False
        # FIXME Order shouldn't matter, but could here
        if self.members != other.members:
            return False
        if self.lengths != other.lengths:
            return False
        return True

_PositionAuctionOutcome = namedtuple(
    "PositionAuctionOutcome", ["allocation", "ppcs"])
_NoExternalityOutcome = namedtuple(
    "PositionAuctionOutcome", ["my_position", "my_ppc"])
_NoExternalityType = namedtuple(
    "NoExternalityType", ["value", "ctr", "quality"])


class NoExternalitySetting(posec.ProjectedSetting):

    def __init__(self, valuations, ctrs, qualities):
        self.n = len(valuations)
        allocations = Permutations(range(self.n))
        payments = posec.RealSpace(self.n)
        self.O = posec.CartesianProduct(
            allocations, payments, memberType=_PositionAuctionOutcome)
        self.Theta = []
        for i in range(self.n):
            self.Theta.append(
                _NoExternalityType(tuple(valuations[i]), tuple(ctrs[i]), qualities[i]))
        self.Psi = posec.CartesianProduct(
            [None] + range(self.n), posec.RealSpace(), memberType=_NoExternalityOutcome)

    def pi(self, i, o):
        my_position = None
        if i in o.allocation:
            my_position = o.allocation.index(i)
        return _NoExternalityOutcome(my_position, o.ppcs[i])

    def u(self, i, theta, po, a_i):
        p = po.my_position
        if p is None:
            return 0.0
        ppc = po.my_ppc
        ctr = theta[i].ctr[p]
        v = theta[i].value[p]
        return ctr * (v - ppc)

_WeightedBid = namedtuple("WeightedBid", ["bid", "effective_bid"])



class NoExternalityPositionAuction(posec.ProjectedMechanism):

    def __init__(self, reserve=1, squashing=1.0, reserveType="UWR", rounding=None, tieBreaking="Uniform", pricing="GSP"):
        self.reserve = reserve
        self.squashing = squashing
        assert reserveType in ["UWR", "QWR"]
        self.reserveType = reserveType
        self.rounding = rounding
        assert tieBreaking in ["Uniform", "Lexigraphic"]
        self.tieBreaking = tieBreaking
        assert pricing in ["GSP", "GFP", "Anchor"]
        if pricing == "Anchor":
            assert reserveType == "UWR"
        self.pricing = pricing

    def q(self, theta_i):
        return math.pow(theta_i.quality, self.squashing)

    def reservePPC(self, i, theta_i):
        if self.reserveType == "UWR":
            return self.reserve
        return self.reserve / self.q(theta_i)

    def makeBid(self, theta_i, b, eb):
        return _WeightedBid(b, eb)

    def A(self, setting, i, theta_i):
        if isinstance(theta_i.value, tuple):
            maxBid = int(math.ceil(max(theta_i.value)))
        else:
            maxBid = int(math.ceil(theta_i.value))
        minBid = int(math.ceil(self.reservePPC(i, theta_i)))
        bids = range(minBid, maxBid + 1)
        if 0 not in bids:
            bids = [0] + bids
        if self.pricing == "Anchor":
            return [self.makeBid(theta_i, b, self.q(theta_i) * (b - self.reserve)) for b in bids]
        else:
            return [self.makeBid(theta_i, b, self.q(theta_i) * b) for b in bids]

    def projectedAllocations(self, i, theta_i, a_N):
        ''' Returns a list of positions (all the positions that can happen depending on how tie-breaking goes)
        the last element in the list means losing every tie-break '''
        higher = a_N.sum(
            [a for a in a_N.actions if a.effective_bid > a_N[i].effective_bid])
        if self.tieBreaking == "Uniform":
            ties = a_N.sum(
                [a for a in a_N.actions if a.effective_bid == a_N[i].effective_bid])
            return range(higher, higher + ties)
        for j in range(i):
            if a_N.plays([a for a in a_N.actions if a.effective_bid == a_N[i].effective_bid]):
                higher += 1
            return [higher]

    def ppc(self, i, theta_i, a_N):
        if self.pricing == "GFP":
            return a_N[i].bid
        if self.pricing == "GSP":
            nextbid = a_N.max(
                [a for a in a_N.actions if a.effective_bid < a_N[i].effective_bid], lambda x: a.effective_bid)
            if nextbid is None:
                return self.reservePPC(i, theta_i)
            ppc = max([nextbid / self.q(theta_i), self.reservePPC(i, theta_i)])
            # FIXME: Rounding
            return ppc
        if self.pricing == "Anchor":
            nextbid = a_N.max(
                [a for a in a_N.actions if a.effective_bid < a_N[i].effective_bid], lambda x: a.effective_bid)
            if nextbid is None:
                return self.reserve
            ppc = nextbid / self.q(theta_i) + self.reserve
            # FIXME: Rounding
            return ppc

    def makeOutcome(self, alloc, price):
        return _NoExternalityOutcome(alloc, price)

    def M(self, setting, i, theta_i, a_N):
        projectedAllocations = self.projectedAllocations(i, theta_i, a_N)
        o = []
        ppc = self.ppc(i, theta_i, a_N)
        for p in projectedAllocations:
            if p == projectedAllocations[-1]:
                o.append(self.makeOutcome(p, ppc))
            else:
                o.append(self.makeOutcome(p, a_N[i].bid))
        return posec.UniformDistribution(o)


def normalizeSetting(setting, k):
    ''' Normalizes values and CTR so that the highest valuation is exactly k and the highest ctr is exactly 1.
Makes to change to quality scores. '''
    maxV = 0.0
    maxCTR = 0.0
    for t in setting.Theta:
        maxV = max([maxV] + list(t.value))
        maxCTR = max([maxCTR] + list(t.ctr))

    # Weird edge-case: Don't do anything if all values/CTRs are zero (or
    # negative)
    if maxV == 0.0:
        maxV = 1.0
    if maxCTR == 0.0:
        maxCTR = 1.0

    for i in range(len(setting.Theta)):
        t = setting.Theta[i]
        value = [v * k / maxV for v in t.value]
        quality = t.quality
        ctr = [c / maxCTR for c in t.ctr]
        setting.Theta[i] = _NoExternalityType(
            tuple(value), tuple(ctr), quality)


def makeAlpha(n, m):
    alphaPrime = 1.0
    alpha = []
    for i in range(m):
            a = alphaPrime * r.random()
            alpha.append(a)
            alphaPrime = a
    for i in range(n - m):
        alpha.append(0.0)
    return alpha


def makeLNAlpha(n, m):
    a = 1.0 + 0.5 * r.random()  # Polynomial decay factor
    alpha = []
    for i in range(m):
        alpha.append(math.pow(i + 1, -a))
    for i in range(n - m):
        alpha.append(0.0)
    return alpha


def EOS(n, m, k, seed=None):
    r.seed(seed)
    beta = r.random()
    values = []
    ctrs = []
    qualities = []
    alpha = makeAlpha(n, m)
    for i in range(n):
        value = r.random() * k
        values.append(tuple([value] * n))
        ctrs.append(tuple([alpha[i] * beta for i in range(n)]))
        qualities.append(beta)
    return NoExternalitySetting(values, ctrs, qualities)


def Varian(n, m, k, seed=None):
    r.seed(seed)
    alpha = makeAlpha(n, m)
    values = []
    ctrs = []
    qualities = []
    for i in range(n):
        beta = r.random()
        value = r.random() * k
        values.append(tuple([value] * n))
        ctrs.append(tuple([alpha[i] * beta for i in range(n)]))
        qualities.append(beta)
    return NoExternalitySetting(values, ctrs, qualities)


def BHN(n, m, k, seed=None):
    r.seed(seed)
    alpha = [1.0]
    convAlpha = [1.0]
    for i in range(1, m):
        alpha.append(r.random() * alpha[i - 1])
        maxIncrease = alpha[i - 1] * convAlpha[
            i - 1] / alpha[i] - convAlpha[i - 1]
        convAlpha.append(r.random() * maxIncrease + convAlpha[i - 1])
    for i in range(m, n):
        alpha.append(0.0)
        convAlpha.append(convAlpha[-1])
    for i in range(n - 1):
        assert alpha[i] >= alpha[i + 1]
        assert convAlpha[i] <= convAlpha[i + 1]
        assert alpha[i] * convAlpha[i] >= alpha[i + 1] * convAlpha[i + 1]
    values = []
    ctrs = []
    qualities = []
    for i in range(n):
        beta = r.random()
        v = r.random() * k
        qualities.append(beta)
        ctrs.append(tuple([beta * a for a in alpha]))
        values.append(tuple([v * convAlpha[i] for i in range(n)]))
    return NoExternalitySetting(values, ctrs, qualities)


def __gauss(mu, sigma, x):
    diff = float(x - mu)
    return math.exp(- (diff * diff) / (2 * sigma * sigma)) / (sigma * math.sqrt(2 * math.pi))


def __rankNormalize(r, m):
    return 1.0 - (r) / (m - 1.0)


def BSS(n, m, k, seed=None):
    r.seed(seed)
    C = 1.0
    alpha = [(0.1 * math.pow(2.0 / 3, i / 2.0)) for i in range(m)]
    while len(alpha) < n:
        alpha.append(0.0)
    values = []
    ctrs = []
    qualities = []
    for i in range(n):
        if r.random() >= 0.5:
            mu = r.random() * 0.2 + 0.8
        else:
            mu = r.random() * 0.2 + 0.4
        v = r.random() * k
        value = [
            __gauss(mu, 0.25 * mu, __rankNormalize(i, m)) * v - C for i in range(n)]
        values.append(tuple(value))
        ctrs.append(tuple(alpha))
        qualities.append(1.0)
    return NoExternalitySetting(values, ctrs, qualities)


def BHN_LN(n, m, k, seed=None):
    r.seed(seed)
    alpha = makeLNAlpha(n, m)
    convAlpha = [1.0]
    for i in range(1, m):
        maxIncrease = alpha[i - 1] * convAlpha[
            i - 1] / alpha[i] - convAlpha[i - 1]
        convAlpha.append(r.random() * maxIncrease + convAlpha[i - 1])
    for i in range(m, n):
        convAlpha.append(convAlpha[-1])
    for i in range(n - 1):
        assert alpha[i] >= alpha[i + 1]
        assert convAlpha[i] <= convAlpha[i + 1]
        assert alpha[i] * convAlpha[i] >= alpha[i + 1] * convAlpha[i + 1]
    values = []
    ctrs = []
    qualities = []
    for i in range(n):
        v, beta = LNdistro(k)
        convBeta = r.random()
        qualities.append(beta)
        ctrs.append(tuple([beta * a for a in alpha]))
        values.append(tuple([v * convAlpha[i] for i in range(n)]))
    return NoExternalitySetting(values, ctrs, qualities)


def gauss2(rho):
    ''' Sample from a 2D Gaussian (with correlation rho) '''
    mu = 0.0, 0.0
    sigma = ((1.0, rho), (rho, 1.0))
    assert len(mu) == 2
    assert len(sigma) == 2
    x0 = r.gauss(mu[0], sigma[0][0])
    mu1 = mu[1] + (x0 - mu[0]) * sigma[1][0] / sigma[0][0]
    sigma1 = sigma[1][1] - sigma[1][0] * sigma[0][1] / sigma[0][0]
    x1 = r.gauss(mu1, sigma1)
    return x0, x1

LN_CORR = 0.4
try:
    LN_params = map(float, string.split(open("LN_params.txt", "r").read()))
except:
    print "WARNING: No parameters for lognormal file found!"
    LN_params = [0.0, 1.0, 0.0, 1.0]


def LNdistro(k):
    global LN_CORR, LN_params
    x0, x1 = gauss2(LN_CORR)
    bid = math.exp((x0 * LN_params[1]) + LN_params[0])
    quality = math.exp((x1 * LN_params[3]) - LN_params[2])
    return bid, quality


def LP(n, m, k, seed=None):
    r.seed(seed)
    alpha = makeLNAlpha(n, m)
    values = []
    ctrs = []
    qualities = []
    for i in range(n):
        value, beta = LNdistro(k)
        values.append(tuple([value] * n))
        ctrs.append(tuple([alpha[i] * beta for i in range(n)]))
        qualities.append(beta)
    return NoExternalitySetting(values, ctrs, qualities)


def EOS_LN(n, m, k, seed=None):
    r.seed(seed)
    beta = r.random()
    values = []
    ctrs = []
    qualities = []
    alpha = makeLNAlpha(n, m)
    for i in range(n):
        value = LNdistro(k)[0]
        values.append(tuple([value] * n))
        ctrs.append(tuple([alpha[i] * beta for i in range(n)]))
        qualities.append(beta)
    return NoExternalitySetting(values, ctrs, qualities)

GENERATORS = {
    "BHN-LN": BHN_LN, "V-LN": LP, "EOS-LN": EOS_LN, "EOS": EOS, "BHN": BHN, "V": Varian, "BSS": BSS,
}
