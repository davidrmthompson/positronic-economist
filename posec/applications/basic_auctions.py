
import posec
from collections import namedtuple

SingleGoodOutcome = namedtuple("SingleGoodOutcome", ["allocation", "payments"])
ProjectedOutcome = namedtuple("ProjectedOutcome", ["i_win", "my_payment"])

SCALE = 10


class FirstPriceAuction(posec.ProjectedMechanism):

    def __init__(self, scale=SCALE):
        self.scale = scale

    def A(self, setting, i, theta_i):
        return range(self.scale)

    def M(self, setting, i, theta_i, a_N):
        if a_N[i] == 0 or a_N.any([a for a in a_N.actions if a > a_N[i]]):
            return ProjectedOutcome(False, 0.0)
        ties = a_N.count(a_N[i])
        outcomes = [
            ProjectedOutcome(True, a_N[i]), ProjectedOutcome(False, 0.0)]
        return posec.Distribution(outcomes, [1.0 / ties, 1 - 1.0 / ties])


class AllPayAuction(FirstPriceAuction):

    def M(self, setting, i, theta_i, a_N):
        if a_N[i] == 0 or a_N.any([a for a in a_N.actions if a > a_N[i]]):
            return ProjectedOutcome(False, a_N[i])
        ties = a_N.count(a_N[i])
        outcomes = [
            ProjectedOutcome(True, a_N[i]), ProjectedOutcome(False, a_N[i])]
        return posec.Distribution(outcomes, [1.0 / ties, 1 - 1.0 / ties])


def ProjectedBayesianSetting(typeDistros):
    ''' typeDistros is an n-length vector of vectors of floats
    for each of thesis vectors, a value of x in the ith position denotes that an agent has valuation of i with probability x
    '''
    n = len(typeDistros)
    allocations = range(n) + [None]
    payments = posec.RealSpace(n)
    O = posec.CartesianProduct(
        allocations, payments, memberType=SingleGoodOutcome)
    Theta = range(max(map(len, typeDistros)))
    P = [posec.Distribution(range(len(d)), d) for d in typeDistros]

    def u(i, theta, o, a_i):
        if o.i_win:
            return theta[i] - o.my_payment
        return -o.my_payment
    Psi = posec.CartesianProduct(
        [True, False], posec.RealSpace(), memberType=ProjectedOutcome)

    def pi(i, o):
        i_win = o.allocation == i
        my_payment = o.payments[i]
        return ProjectedOutcome(i_win, my_payment)
    return posec.ProjectedBayesianSetting(n, O, Theta, P, u, Psi, pi)


def welfareTransform(setting, i, theta_N, o, a_i):
    if o.i_win:
        return theta_N[i]
    return 0.0


def paymentTransform(setting, i, theta_N, o, a_i):
    return o.my_payment
