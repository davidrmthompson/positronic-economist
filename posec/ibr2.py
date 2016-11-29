import logging
import os
import random

import time
from pyagg import _delta, AGG_File

logger = logging.getLogger(__name__)


def random_pure_strategy(agg):
    strategies = []
    for size in agg.aSizes:
        strategy = _delta(size, random.randrange(size))
        strategies.append(strategy)
    return strategies


def idx_maxes(l):
    m = max(l)
    return [i for i,a in enumerate(l) if a == m]


def random_idx_max(l):
    return random.choice(idx_maxes(l))


def IBR(agg, seed=None, output=None, cutoff=3600):
    start = time.time()

    random.seed(seed)

    file = output if output is not None else os.devnull

    with open(file, 'w') as f:
        f.write('max_regret,weighted_max_regret,cputime\n')

        def log_strategy(s):
            max_regret = agg.max_regret(s)
            elapsed = time.time()-start
            weighted_max_regret = max_regret / agg.max_payoff
            f.write(','.join((str(max_regret), str(max_regret/agg.max_payoff), str(elapsed)))+'\n')
            logger.info("Weighted max regret overall is %s. Time elapsed %.2f" % (weighted_max_regret, elapsed))

        strategy = random_pure_strategy(agg)
        iteration = 0
        while not agg.isNE(strategy) and time.time() - start < cutoff:
            log_strategy(strategy)

            # Pick a random player
            player_idx = random.randrange(len(agg.N))

            # Calculate best response for this player - the action with the highest regret (random in ties)
            regrets = agg.regret(strategy, asLL=True)[player_idx]
            best_response = random_idx_max(regrets)

            # Update the strategy profile
            strategy[player_idx] = _delta(agg.aSizes[player_idx], best_response)

            iteration += 1

        log_strategy(strategy)
        # Either cutoff or NE
        if agg.isNE(strategy):
            logger.info("NE FOUND at iteration %d after %.2f" % (iteration, time.time()-start))
            logger.info(strategy)
        else:
            logger.info("Cutoff reached")

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)-15s [%(levelname)s] %(message)s', level=logging.INFO, filename='posec.log')
    logging.getLogger().addHandler(logging.StreamHandler())
    for n in range(1,11):
        path = '/ubc/cs/research/arrow/newmanne/positronic-economist/baggs/GFP/'
        agg = AGG_File(path + 'gfp_10_' + str(n) + '_1_FINAL.bagg')
        IBR(agg, seed=1)