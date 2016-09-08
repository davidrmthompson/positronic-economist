import itertools
import random

import posec
from posec import *
from posec import mathtools
import time
import collections
from posec.applications import position_auctions
from posec.applications.position_auctions import *
import argparse
import redis
import json
import logging

def two_approval(n, m_candidates=5, num_types=6):
    # Outcomes - the candiate that gets elected
    O = tuple("c" + str(i+1) for i in range(m_candidates))

    # Pick some types, since too many is not computationally feasible
    all_possible_rankings = mathtools.permutations(O)
    Theta = [tuple(random.choice(all_possible_rankings)) for _ in range(num_types)]

    # Get probabilities over these types
    P = [UniformDistribution(Theta)] * n

    def u(i, theta, o, a_i):
        return len(O) - theta[i].index(o)

    setting = BayesianSetting(n, O, Theta, P, u)

    def A(setting, i, theta_i):
        return itertools.combinations(setting.O, r=2)

    def M(setting, i, theta_i, a_N):
        scores = {o: a_N.sum([a for a in a_N.actions if a is not None and o in a]) for o in setting.O}
        max_score = max(scores.values())
        winners = [o for o in scores.keys() if scores[o] == max_score]
        return posec.UniformDistribution(winners)

    t0 = time.time()
    agg = makeAGG(setting, ProjectedMechanism(A, M), symmetry=True)
    t1 = time.time()
    return agg.sizeAsAGG(), agg.sizeAsAGG(), t1-t0
    # agg.saveToFile("paper_examples_voting.bagg")

# def two_approval(n, m_candidates=5):
#     # Outcomes - the candiate that gets elected
#     O = tuple("c" + str(i+1) for i in range(m_candidates))
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
#         scores = {}
#         for o in setting.O:
#             scores[o] = a_N.sum([a for a in a_N.actions if a is not None and o in a])
#         max_score = max(scores.values())
#         winners = [o for o in scores.keys() if scores[o] == max_score]
#         return posec.UniformDistribution(winners)
#
#     mechanism = Mechanism(A, M)
#
#     agg = makeAGG(setting, mechanism, symmetry=True)
#     return agg.sizeAsNFG(), agg.sizeAsAGG(), t1-t0


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
    setting = Varian(n, 3, 15, seed)
    m = position_auctions.NoExternalityPositionAuction(pricing="GFP", squashing=0.0)
    return setting, m

def gsp(n, seed=None):
    #n, m, k, seed = None
    setting = Varian(n, 3, 15, seed)
    m = position_auctions.NoExternalityPositionAuction(pricing="GSP", squashing=1.0)
    return setting, m

def bbsi_check(n_players, seed, fn, bbsi_level):
    name = "%s_%d_%d" % (fn.__name__, n_players, seed)
    print name
    setting, m = fn(n_players, seed)
    metrics = dict(n_players=n_players, seed=seed, game=fn.__name__, bbsi_level=bbsi_level)
    agg = makeAGG(setting, m, bbsi_level=2, metrics=metrics)
    agg.saveToFile("baggs/%s/%s.bagg" % (fn.__name__.upper(), name))
    return metrics

def test():
    logging.basicConfig(format='%(asctime)-15s [%(levelname)s] %(message)s', level=logging.INFO, filename='posec.log')
    logging.getLogger().addHandler(logging.StreamHandler())
    setting = Varian(10, 3, 15, 1)
    m = position_auctions.NoExternalityPositionAuction(pricing="GSP", squashing=1.0)
    agg = makeAGG(setting, m, bbsi_level=0)


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
            'vote': None
        }
        job['fn'] = g2f[job['game']]
        logging.info("Running job %s" % job)
        metrics = bbsi_check(job['n'], job['seed'], job['fn'], job['bbsi_level'])
        with open(args.file, "a") as output:
            output.write(json.dumps(metrics)+'\n')
        r.lrem(q + '_PROCESSING', 1, instance)
    print "ALL DONE!"
