import logging
import os
import random

import numpy as np

from posec_core import cputime
from pyagg import fromLLtoString, AGG_File
from ibr2 import random_idx_max

logger = logging.getLogger(__name__)

def uniform_mixed_strategy(agg):
    strategies = []
    for size in agg.aSizes:
        strategy = [1. / size] * size
        strategies.append(strategy)
    return strategies

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    return v/norm

def normalize_model(model):
    return map(normalize, model)

def FP(agg, seed=None, output=None, cutoff=3600):
    start = cputime()

    random.seed(seed)

    file = output if output is not None else os.devnull

    with open(file, 'a') as f:
        f.write('max_regret,cputime' + '\n')

        model = uniform_mixed_strategy(agg)
        strategy = model
        iteration = 0
        while not agg.isNE(strategy) and cputime() - start < cutoff:
            max_regret = agg.max_regret(strategy)
            f.write(str(max_regret) + ',' + str(cputime() - start) + '\n')
            logger.info("Max regret overall is %s" % (max_regret))

            # Get everyone's current regret given the model's count frequencies as empirical mixed strategy
            regrets = agg.regret(strategy, asLL=True)

            # Calculate each player's best response, update that count by 1. Break ties randomly
            for i in range(len(agg.N)):
                best_response = random_idx_max(regrets[i])
                model[i][best_response] += 1

            strategy = normalize_model(model)

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
        FP(agg)