import logging
import os
import random

from posec_core import cputime
from pyagg import _delta, fromLLtoString, AGG_File

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
    start = cputime()

    random.seed(seed)

    file = output if output is not None else os.devnull

    with open(file, 'a') as f:
        f.write('max_regret,cputime' + '\n')

        strategy = random_pure_strategy(agg)
        iteration = 0
        while not agg.isNE(strategy) and cputime() - start < cutoff:
            max_regret = agg.max_regret(strategy)
            f.write(str(max_regret)+','+str(cputime()-start)+'\n')
            logger.info("Max regret overall is %s" % (max_regret))

            # Pick a random player
            player_idx = random.randrange(len(agg.N))

            # Calculate best response for this player - the action with the highest regret (random in ties)
            regrets = agg.regret(strategy, asLL=True)[player_idx]
            best_response = random_idx_max(regrets)

            # Update the strategy profile
            strategy[player_idx] = _delta(agg.aSizes[player_idx], best_response)

            iteration += 1

        # Either cutoff or NE
        if agg.isNE(strategy):
            logger.info("NE FOUND at iteration %d" % iteration)
            logger.info(agg.interpretProfile(fromLLtoString(strategy)))
        else:
            logger.info("Cutoff reached")

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)-15s [%(levelname)s] %(message)s', level=logging.INFO, filename='posec.log')
    logging.getLogger().addHandler(logging.StreamHandler())
    for n in range(1,11):
        path = '/ubc/cs/research/arrow/newmanne/positronic-economist/baggs/GSP/'
        agg = AGG_File(path + 'gsp_10_' + str(n) + '_1_FINAL.bagg')
        IBR(agg)