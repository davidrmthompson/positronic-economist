from collections import namedtuple, Counter
from posec import mathtools
from position_auctions import Permutations, _PositionAuctionOutcome, NoExternalityPositionAuction
import math
import posec
import string

# FIXME: I'm overloading externality type as a word. I want something more general than continuation probability
# (to allow for the rich externality model), but that's not it.

_ExternalityOutcome = namedtuple(
    "ExternalityOutcome", ["higher_externality_types", "my_ppc"])
_ExternalityType = namedtuple(
    "ExternalityType", ["value", "ctr", "quality", "externality_type"])
_ExternalityWeightedBid = namedtuple(
    "WeightedBid", ["bid", "effective_bid", "externality_type"])


class HybridSetting(posec.ProjectedSetting):

    def __init__(self, valuations, ctrs, qualities, continuation_probabilities):
        self.n = len(valuations)
        projectedAllocations = mathtools.powerSet(
            continuation_probabilities, Counter)
        allocations = Permutations(range(self.n))
        payments = posec.RealSpace(self.n)
        self.O = posec.CartesianProduct(
            allocations, payments, memberType=_PositionAuctionOutcome)
        self.Theta = []
        for i in range(self.n):
            self.Theta.append(
                _ExternalityType(tuple(valuations[i]), tuple(ctrs[i]), qualities[i], continuation_probabilities[i]))
        self.Psi = posec.CartesianProduct(
            projectedAllocations, posec.RealSpace(), memberType=_ExternalityOutcome)

    def ctr(self, i, theta, projectedAllocation):
        return mathtools.product(projectedAllocation) * theta[i].ctr[len(projectedAllocation)]

    def u(self, i, theta, po, a_i):
        projectedAllocation = po.higher_externality_types
        if projectedAllocation == None:
            return 0.0
        p = len(projectedAllocation)
        ppc = po.my_ppc
        ctr = self.ctr(i, theta, projectedAllocation)
        v = theta[i].value[p]
        return ctr * (v - ppc)


class ExternalityPositionAuction(NoExternalityPositionAuction):

    def makeOutcome(self, alloc, price):
        return _ExternalityOutcome(alloc, price)

    def makeBid(self, theta_i, b, eb):
        return _ExternalityWeightedBid(b, eb, theta_i.externality_type)

    def projectedAllocations(self, i, theta_i, a_N):
        ''' Returns a list of all the projected allocations (a projected allocation is the list of externality types of agents ranked above i) '''
        possible_externality_types = set(
            [theta_j.externality_type for theta_j in a_N.types])
        higher_externality_types = Counter()
        for theta_j in possible_externality_types:
            m = a_N.sum([a for a in a_N.actions if a.effective_bid > a_N[
                        i].effective_bid and a.externality_type == theta_j])
            higher_externality_types[theta_j] += m
        if self.tieBreaking == "Uniform":
            equal_externality_types = Counter()
            for theta_j in possible_externality_types:
                m = a_N.sum(
                    [a for a in a_N.actions if a.effective_bid == a_N[i].effective_bid and a.externality_type == theta_j])
                if theta_j == theta_i.externality_type:
                    m -= 1
                equal_externality_types[theta_j] += m
            print equal_externality_types
            subsets = mathtools.powerSet(
                list(equal_externality_types.elements()), Counter)
            return [higher_externality_types + subset for subset in subsets]
        raise BaseException("Lexigraphic tie-breaking is not working yet")

# GENERATOR FUNCTIONS ####


class _CascadeOrHybridFactory:

    def __init__(self, alpha_function, value_genrator_function):
        self.alpha_function = alpha_function
        self.value_genrator_function = value_genrator_function

    def __call__(self, n, m, k, seed=None):
        import random as r
        r.seed(seed)
        valuations = []
        ctrs = []
        qualities = []
        continuation_probabilities = []
        for i in range(n):
            value, quality = self.value_genrator_function(r, k)
            valuations.append([value] * n)
            qualities.append(quality)
            ctrs.append(
                [quality * alpha for alpha in self.alpha_function(n, m, r)])
            continuation_probabilities.append(r.random())
        return HybridSetting(valuations, ctrs, qualities, continuation_probabilities)


def _uni_distro(r, k):
    return r.random() * k, r.random()


def _make_flat_alpha(n, m, r):
    return [1.0] * m + [1.0] * (n - m)


def _make_UNI_alpha(n, m, r):
    alphaPrime = 1.0
    alpha = []
    for i in range(m):
            a = alphaPrime * r.random()
            alpha.append(a)
            alphaPrime = a
    for i in range(n - m):
        alpha.append(0.0)
    return alpha


def _make_LP_alpha(n, m, r):
    a = 1.0 + 0.5 * r.random()  # Polynomial decay factor
    alpha = []
    for i in range(m):
        alpha.append(math.pow(i + 1, -a))
    for i in range(n - m):
        alpha.append(0.0)
    return alpha

from posec.applications.position_auctions import LNdistro as _ln_distro

cascade_UNI = _CascadeOrHybridFactory(_make_flat_alpha, _uni_distro)
cascade_LN = _CascadeOrHybridFactory(_make_flat_alpha, _ln_distro)
hybrid_UNI = _CascadeOrHybridFactory(_make_UNI_alpha, _uni_distro)
hybrid_LN = _CascadeOrHybridFactory(_make_LP_alpha, _ln_distro)

GENERATORS = {"cascade_UNI": cascade_UNI, "cascase_LN":
              cascade_LN, "hybrid_UNI": hybrid_UNI, "hybrid_LN": hybrid_LN}

# def richExternality(n,m,k,seed):
#     import random as r
#     setting = CascadeSetting(n,m,k)
#     setting.T = []
#     r.seed(seed)
#     print seed,r.random()
#     for i in range(n):
#         t = RichExternalityType(setting)
#         beta = r.random()
#         t.value = r.random()*k
#         print beta,t.value
#         t.q = beta
#         t.advertiserCTRFactor = beta
#         t.ctrDict = richExternalityDictionary(n,r)
#         setting.T.append(t)
#         print t
#     return setting

# def richExternality_LN(n,m,k,seed):
#     import random as r
#     setting = CascadeSetting(n,m,k)
#     setting.T = []
#     r.seed(seed)
#     for i in range(n):
#         t = RichExternalityType(setting)
#         t.value, beta = LNdistro(r,k)
#         t.q = beta
#         t.advertiserCTRFactor = beta
#         setting.T.append(t)
#         t.ctrDict = richExternalityDictionary(n,r)
#     return setting


# GENERATORS = {
#     "BHN-LN":BHN_LN,
#     "CAS-LN":cascade_LN,"V-LN":LP,"BSS-LN":BSS_LN,"EOS-LN":EOS_LN,"EOS":EOS,"BHN":BHN,"V":Varian,"BSS":BSS,"CAS":CascadeSetting,
#     "HYB-LN":hybrid_LN,"HYB":hybrid,"RE":richExternality, "RE-LN":richExternality_LN
#     }
