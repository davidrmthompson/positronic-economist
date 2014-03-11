import posec
from posec import pyagg
import unittest

from posec import *


class PaperExamples(unittest.TestCase):

    def test_voting(self):
        n = 10
        O = ("c1", "c2", "c3")
        Theta = [("c1", "c2", "c3"),
                 ("c2", "c3", "c1"),
                 ("c3", "c1", "c2"),
                 ("c1", "c3", "c2")]
        P = [UniformDistribution(Theta)] * n

        def u(i, theta, o, a_i):
            return theta[i].index(o)
        setting = BayesianSetting(n, O, Theta, P, u)

        def A(setting, i, theta_i):
            return setting.O

        def M(setting, a_N):
            for c1 in setting.O:
                c1Wins = True
                for c2 in setting.O:
                    if a_N.count(c2) > a_N.count(c1):
                        c1Wins = False
                if c1Wins:
                    return c1
        mechanism = Mechanism(A, M)
        agg = makeAGG(setting, mechanism, symmetry=True)
        agg.saveToFile("paper_examples_voting.bagg")

    def test_single_good_auction(self):
        n = 2
        allocations = range(n) + [None]
        payments = RealSpace(n)
        O = CartesianProduct(allocations, payments)
        Theta = [1, 2, 3, 4]
        P = [
            Distribution(support=[1, 2, 3, 4],
                         probabilities=[0.2, 0.3, 0.1, 0.4]),
            Distribution(support=[1, 2, 3, 4],
                         probabilities=[0.4, 0.3, 0.2, 0.1])
        ]

        def u(i, theta, o, a_i):
            alloc, payments = o
            if alloc == i:
                return theta[i] - payments[i]
            return -payments[i]
        setting = BayesianSetting(n, O, Theta, P, u)

        def A(setting, i, theta_i):
            return [0, 1, 2, 3, 4]

        def M(setting, a_N):
            highBid = a_N.max(a_N.actions)
            winners = [i for i in range(setting.n)
                       if a_N[i] == highBid]
            payments = [a_N[i] for i in range(setting.n)]
            outcomes = [(i, payments) for i in winners]
            return UniformDistribution(outcomes)
        mechanism = Mechanism(A, M)
        agg = makeAGG(setting, mechanism)
        agg.saveToFile("paper_examples_auction.bagg")

    def test_projected_auction(self):
        n = 2
        allocations = range(n) + [None]
        payments = RealSpace(n)
        O = CartesianProduct(allocations, payments)
        Theta = [1, 2, 3, 4]
        P = [
            Distribution(support=[1, 2, 3, 4],
                         probabilities=[0.2, 0.3, 0.1, 0.4]),
            Distribution(support=[1, 2, 3, 4],
                         probabilities=[0.4, 0.3, 0.2, 0.1])
        ]

        Psi = CartesianProduct([True, False], RealSpace())

        def pi(i, o):
            alloc, payments = o
            return alloc == i, payments[i]

        def u(i, theta, psi, a_i):
            i_win, my_payment = psi
            if i_win:
                return theta[i] - my_payment
            return -my_payment
        setting = ProjectedBayesianSetting(n, O, Theta, P, u, Psi, pi)

        def A(setting, i, theta_i):
            return [0, 1, 2, 3, 4]

        def M(setting, i, theta_i, a_N):
            bid = a_N[i]
            higherBids = [a for a in a_N.actions if a > bid]
            if a_N.any(higherBids):
                return (False, bid)
            ties = a_N.count(bid)
            return Distribution(support=[(True, bid), (False, bid)],
                                probabilities=[1.0 / ties, 1.0 - 1.0 / ties])
        mechanism = ProjectedMechanism(A, M)
        agg = makeAGG(setting, mechanism, symmetry=True)
        agg.saveToFile("paper_examples_projected_auction.bagg")


class ApplicationExamples(unittest.TestCase):

    def test_basic_auction(self):
        from posec.applications import basic_auctions
        distros = [
            [0.25] * 4 + [0.0],
            [0.2] * 5
        ]
        setting = basic_auctions.ProjectedBayesianSetting(distros)
        auction = basic_auctions.FirstPriceAuction(5)
        agg = makeAGG(setting, auction, symmetry=True,
                      transform=basic_auctions.welfareTransform)
        agg.saveToFile("applications_basic_auction.welfare_bagg")

        auction = basic_auctions.AllPayAuction(5)
        agg = makeAGG(setting, auction, symmetry=True,
                      transform=basic_auctions.paymentTransform)
        agg.saveToFile("applications_basic_auction.payment_bagg")

    def test_voting(self):
        from posec.applications import voting
        setting = voting.uniform_Setting(4, 3, 1)
        for mech_class in voting.MECHANISM_CLASSES:
            args = [2] if mech_class.__name__ == "kApproval" else []
            mechanism = mech_class(*args)
            agg = makeAGG(setting, mechanism, symmetry=True)
            agg.saveToFile("applications_voting_%s.agg" %
                           (mech_class.__name__,))

    def test_position_auctions(self):
        from posec.applications import position_auctions
        mechanisms = {
            "GFP": position_auctions.NoExternalityPositionAuction(
                pricing="GFP", squashing=0.0),
            "wGSP": position_auctions.NoExternalityPositionAuction(
                pricing="GSP", squashing=1.0),
            "uGSP": position_auctions.NoExternalityPositionAuction(
                pricing="GSP", squashing=0.0)
        }
        for setting_name, setting_fn in position_auctions.GENERATORS.items():
            setting = setting_fn(3, 2, 8, 1)
            for mech_name, mech in mechanisms.items():
                agg = makeAGG(setting, mech)
                agg.saveToFile("applications_%s_%s.agg" %
                               (mech_name, setting_name))

    def test_position_auctions_externalities(self):
        from posec.applications import position_auctions_externalities
        V = [[1, 1], [2, 2]]
        CTRs = [[1.0, 0.5], [1.0, 0.1]]
        qualities = [1.0, 1.0]
        continuations = [0.25, 0.5]

        setting = position_auctions_externalities.HybridSetting(
            V, CTRs, qualities, continuations)
        mech = position_auctions_externalities.ExternalityPositionAuction(
            pricing='GFP', tieBreaking="Lexigraphic")
        agg = makeAGG(setting, mech)
        agg.saveToFile("applications_GFP_hybrid.agg")
if __name__ == '__main__':
    unittest.main()
