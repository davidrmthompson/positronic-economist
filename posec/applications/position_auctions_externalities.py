
from collections import namedtuple
import posec
from posec import mathtools

from position_auctions import Permutations, _PositionAuctionOutcome, NoExternalityPositionAuction

_ExternalityOutcome = namedtuple(
    "ExternalityOutcome", ["higher_externality_types", "my_ppc"])
_ExternalityType = namedtuple(
    "ExternalityType", ["value", "ctr", "quality", "externality_type"])
_ExternalityWeightedBid = namedtuple(
    "WeightedBid", ["bid", "effective_bid", "externality_type"])


class HybridSetting(posec.ProjectedSetting):

    def __init__(self, valuations, ctrs, qualities, continuation_probabilities):
        self.n = len(valuations)
        projectedAllocations = mathtools.powerSet(continuation_probabilities)
        allocations = Permutations(range(self.n))
        payments = posec.RealSpace(self.n)
        self.O = posec.CartesianProduct(
            allocations, payments, memberType=_PositionAuctionOutcome)
        self.Theta = []
        for i in range(self.n):
            self.Theta.append(
                _ExternalityType(tuple(valuations[i]), tuple(ctrs[i]), qualities[i], continuation_probabilities[i]))
        projectedAllocations = mathtools.powerSet(continuation_probabilities)
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
        higher_externality_types = []
        for theta_j in possible_externality_types:
            m = a_N.sum([a for a in a_N.actions if a.effective_bid > a_N[
                        i].effective_bid and a.externality_type == theta_j])
            higher_externality_types += [theta_j] * m
        if self.tieBreaking == "Uniform":
            equal_externality_types = []
            for theta_j in possible_externality_types:
                m = a_N.sum(
                    [a for a in a_N.actions if a.effective_bid == a_N[i].effective_bid and a.externality_type == theta_j])
                if theta_j == theta_i.externality_type:
                    m -= 1
                equal_externality_types += [theta_j] * m
            return [higher_externality_types + subset for subset in mathtools.powerSet(equal_externality_types)]
        raise BaseException("Lexigraphic tie-breaking is not working yet")
