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
from posec.ibr2 import IBR
from posec.fp2 import FP
from posec.pyagg import AGG_File, gnm

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

def two_approval_setting(n, seed=None):
    random.seed(seed)
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
    return list(itertools.combinations(setting.O, r=2))

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

def bad_gfp(n, seed=None):
    setting = Varian(n, N_POSITIONS, N_BIDS, seed)
    m = position_auctions.NoExternalityPositionAuction(pricing="GFP", squashing=0.0, bad=True)
    return setting, m

def bad_gsp(n, seed=None):
    setting = Varian(n, N_POSITIONS, N_BIDS, seed)
    m = position_auctions.NoExternalityPositionAuction(pricing="GSP", squashing=1.0, bad=True)
    return setting, m

def gfp(n, seed=None):
    setting = Varian(n, N_POSITIONS, N_BIDS, seed)
    m = position_auctions.NoExternalityPositionAuction(pricing="GFP", squashing=0.0)
    return setting, m

def gsp(n, seed=None):
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
    bagg_filedir = "baggs/%s" % fn.__name__.upper()
    if not os.path.exists(bagg_filedir):
        os.makedirs(bagg_filedir)
    metrics['dir'] = bagg_filedir
    metrics['name'] = name

    agg = makeAGG(setting, m, bbsi_level=bbsi_level, metrics=metrics, symmetry=symmetry)
    # explain(agg)
    agg.saveToFile("%s/%s_FINAL.bagg" % (bagg_filedir, name))
    return metrics


def test():
    path = '/ubc/cs/research/arrow/newmanne/positronic-economist/baggs/GSP/'
    agg = AGG_File(path + 'gsp_2_1_1_FINAL.bagg')
    GNM(agg)

def GNM(agg, seed=None, output=None, cutoff=3600):
    file = output if output is not None else os.devnull
    with open(file, 'a') as f:
        extra_args = {}
        if seed is not None:
            extra_args['seed'] = seed
        gnm.solve(agg, **extra_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qname', type=str, help="redis queue")
    parser.add_argument('--host', type=str, help="redis host")
    parser.add_argument('--port', type=int, help="redis port")
    parser.add_argument('--file', type=str, help="output file")
    parser.add_argument('--logfile', type=str, help="log file", default="posec.log")
    args = parser.parse_args()

    if not (args.qname and args.host and args.port):
        test()
    else:
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
            if 'agg_file' in job:
                alg2f = {
                    'IBR': IBR,
                    'FP': FP,
                    'GNM': GNM
                }
                # TODO: add GNM
                alg2f[job['alg']](AGG_File(job['agg_file']), seed=job['alg_seed'], output=job['output'], cutoff=job['cutoff'])
            else:
                g2f = {
                    'GFP': gfp,
                    'GFP_bad': bad_gfp,
                    'GSP': gsp,
                    'GSP_bad': bad_gsp,
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
