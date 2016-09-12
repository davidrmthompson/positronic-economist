import itertools
import logging
import random

from posec import *
from posec import mathtools
from posec.applications import position_auctions
from posec.applications.position_auctions import *
from collections import defaultdict
import argparse
import json
import redis
import os

N_POSITIONS = 4
N_BIDS = 20


# def two_approval(n, m_candidates=5, num_types=6):
#     # Outcomes - the candiate that gets elected
#     O = tuple("c" + str(i+1) for i in range(m_candidates))
#
#     # Pick some types, since too many is not computationally feasible
#     all_possible_rankings = mathtools.permutations(O)
#     Theta = [tuple(random.choice(all_possible_rankings)) for _ in range(num_types)]
#
#     # Get probabilities over these types
#     P = [UniformDistribution(Theta)] * n
#
#     def u(i, theta, o, a_i):
#         return len(O) - theta[i].index(o)
#
#     setting = BayesianSetting(n, O, Theta, P, u)
#
#     def A(setting, i, theta_i):
#         return itertools.combinations(setting.O, r=2)
#
#     def M(setting, i, theta_i, a_N):
#         scores = {o: a_N.sum([a for a in a_N.actions if a is not None and o in a]) for o in setting.O}
#         max_score = max(scores.values())
#         winners = [o for o in scores.keys() if scores[o] == max_score]
#         return posec.UniformDistribution(winners)
#
#     t0 = time.time()
#     agg = makeAGG(setting, ProjectedMechanism(A, M), symmetry=True)
#     t1 = time.time()
#     return agg.sizeAsAGG(), agg.sizeAsAGG(), t1-t0
    # agg.saveToFile("paper_examples_voting.bagg")

def two_approval_setting(n):
    m_candidates = 5
    # Outcomes - the candiate that gets elected
    O = tuple("c" + str(i + 1) for i in range(m_candidates))

    # Pick some types
    all_possible_rankings = mathtools.permutations(O)
    Theta = [tuple(random.choice(all_possible_rankings)) for _ in range(n)]

    def u(i, theta, o, a_i):
        return theta[i].index(o)

    return Setting(n, O, Theta, u)


def two_approval_A(setting, i, theta_i):
    return itertools.combinations(setting.O, r=2)

def two_approval(n, seed=None):
    random.seed(seed)
    setting = two_approval_setting(n)

    def M(setting, a_N):
        scores = {}
        for o in setting.O:
            scores[o] = a_N.sum([a for a in a_N.actions if o in a])
        max_score = max(scores.values())
        winners = [o for o in scores.keys() if scores[o] == max_score]
        return posec.UniformDistribution(winners)
    mechanism = Mechanism(two_approval_A, M)

    return setting, mechanism

def bad_two_approval(n, seed=None):
    random.seed(seed)
    setting = two_approval_setting(n)

    # Poorly specified version!
    def M(setting, a_N):
        scores = defaultdict(int)
        for i in range(setting.n):
            action = a_N[i]
            for vote in action:
                scores[vote] += 1
        max_score = max(scores.values())
        winners = [o for o in scores.keys() if scores[o] == max_score]
        return posec.UniformDistribution(winners)

    mechanism = Mechanism(two_approval_A, M)
    return setting, mechanism

# def bad_gfp(n, seed=None):
#     r.seed(seed)
#     m = N_POSITIONS
#     k = N_BIDS
#
#     _PositionAuctionOutcome = namedtuple(
#         "PositionAuctionOutcome", ["allocation", "ppcs"])
#     _NoExternalityType = namedtuple(
#         "NoExternalityType", ["value", "ctr", "quality"])
#     _WeightedBid = namedtuple("WeightedBid", ["bid", "effective_bid"])
#
#     alpha = makeAlpha(n, m)
#     values = []
#     ctrs = []
#     qualities = []
#     for i in range(n):
#         beta = r.random()
#         value = r.random() * k
#         values.append(tuple([value] * n))
#
#         ctrs.append(tuple([alpha[i] * beta for i in range(n)]))
#         qualities.append(beta)
#
#         n = len(values)
#         allocations = Permutations(range(n))
#         payments = posec.RealSpace(n)
#         O = posec.CartesianProduct(
#             allocations, payments, memberType=_PositionAuctionOutcome)
#         Theta = []
#         for i in range(n):
#             Theta.append(
#                 _NoExternalityType(tuple(values[i]), tuple(ctrs[i]), qualities[i]))
#
#     def u(i, theta, o, a_i):
#         if i not in o.allocation:
#             return 0.0
#         else:
#             pos = o.allocation.index(i)
#             ppc = o.ppcs[i]
#             ctr = theta[i].ctr[pos]
#             v = theta[i].value[pos]
#             return ctr * (v - ppc)
#
#
#     # TODO: GSP
#     def q(a):
#         return 1
#
#     setting = Setting(n, O, Theta, u)
#
#     def A(self, setting, i, theta_i):
#         # Exclude the obviously dominated strategy of bidding above your value
#         max_bid = int(math.ceil(theta_i.value))
#         min_bid = 1
#         bids = range(min_bid, max_bid + 1)
#         if 0 not in bids:
#             bids = [0] + bids
#         return [_WeightedBid(theta_i, b, q(theta_i) * b) for b in bids]
#
#     def M(setting, a_N):
#         acts = []
#         for i in range(setting.n):
#             bid = a_N[i].effective_bid
#             acts.append({'id': i, 'bid': bid})
#         acts.sort(key=lambda x: x.bid)
#         allocations = []
#         payments = []
#         for a in acts:
#
#
#
#         return posec.UniformDistribution(winners)
#
#     mechanism = Mechanism(A, M)
#     return setting, mechanism

# def slopppy_two_approval(n, m_candidates=5):
#     # Outcomes - the candiate that gets elected
#     O = tuple("c" + str(i + 1) for i in range(m_candidates))
#
#     # Pick some types
#     all_possible_rankings = mathtools.permutations(O)
#     Theta = [tuple(random.choice(all_possible_rankings)) for _ in range(n)]
#
#     def u(i, theta, o, a_i):
#         return len(O) - theta[i].index(o)
#
#     setting = Setting(n, O, Theta, u)
#
#     def A(setting, i, theta_i):
#         return itertools.combinations(setting.O, r=2)
#
#     def M(setting, a_N):
#         scores = collections.defaultdict(int)
#         n = setting.n
#         for i in range(n):
#             action_i = a_N[i]
#             o1, o2 = action_i
#             scores[o1] += 1
#             scores[o2] += 1
#         max_score = max(scores.values())
#         winners = [o for o in scores.keys() if scores[o] == max_score]
#         return posec.UniformDistribution(winners)
#
#     mechanism = Mechanism(A, M)
#
#     t0 = time.time()
#     agg = makeAGG(setting, mechanism, symmetry=False)#, bbsi_level=2)
#     t1 = time.time()
#     if n == 2:
#         agg.saveToFile("wbsi_bad.bagg")
#
#     return agg.sizeAsNFG(), agg.sizeAsAGG(), t1 - t0


def gfp(n, seed=None):
    #n, m, k, seed = None
    setting = Varian(n, N_POSITIONS, N_BIDS, seed)
    m = position_auctions.NoExternalityPositionAuction(pricing="GFP", squashing=0.0)
    return setting, m

def gsp(n, seed=None):
    #n, m, k, seed = None
    setting = Varian(n, N_POSITIONS, N_BIDS, seed)
    m = position_auctions.NoExternalityPositionAuction(pricing="GSP", squashing=1.0)
    return setting, m

def bbsi_check(n_players, seed, fn, bbsi_level):
    name = "%s_%d_%d_%d" % (fn.__name__, n_players, seed, bbsi_level)
    print name
    setting, m = fn(n_players, seed=seed)
    metrics = dict(n_players=n_players, seed=seed, game=fn.__name__, bbsi_level=bbsi_level)
    symmetry = True if fn == two_approval else False
    print "Symmetry", symmetry
    agg = makeAGG(setting, m, bbsi_level=bbsi_level, metrics=metrics, symmetry=symmetry)
    bagg_filedir = "baggs/%s" % fn.__name__.upper()
    if not os.path.exists(bagg_filedir):
        os.makedirs(bagg_filedir)
    agg.saveToFile("%s/%s.bagg" % (bagg_filedir, name))
    return metrics

def test():
    logging.basicConfig(format='%(asctime)-15s [%(levelname)s] %(message)s', level=logging.INFO, filename='posec.log')
    logging.getLogger().addHandler(logging.StreamHandler())
    bbsi_check(4,1,gfp,1)
    # setting = Varian(10, 3, 15, 1)
    # m = position_auctions.NoExternalityPositionAuction(pricing="GSP", squashing=1.0)
    # agg = makeAGG(setting, m, bbsi_level=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qname', type=str, help="redis queue", required=True)
    parser.add_argument('--host', type=str, help="redis host", required=True)
    parser.add_argument('--port', type=int, help="redis port", required=True)
    parser.add_argument('--file', type=str, help="output file", required=True)
    parser.add_argument('--logfile', type=str, help="log file", default="posec.log")
    args = parser.parse_args()
    r = redis.StrictRedis(host=args.host, port=args.port)
    q = args.qname

    logging.basicConfig(format='%(asctime)-15s [%(levelname)s] %(message)s', level=logging.INFO, filename=args.logfile)
    logging.getLogger().addHandler(logging.StreamHandler())

    while True:
        remaining_jobs = r.llen(q)
        logging.info("There are %d jobs remaining" % (remaining_jobs))
        instance = r.rpoplpush(q, q + '_PROCESSING')
        if instance is None:
            break
        job = json.loads(instance)
        g2f = {
            'GFP': gfp,
            'GSP': gsp,
            'vote': two_approval,
            'vote_bad': bad_two_approval
        }
        job['fn'] = g2f[job['game']]
        logging.info("Running job %s" % job)
        metrics = bbsi_check(job['n'], job['seed'], job['fn'], job['bbsi_level'])
        with open(args.file, "a") as output:
            output.write(json.dumps(metrics)+'\n')
        r.lrem(q + '_PROCESSING', 1, instance)
    print "ALL DONE!"
