import logging
import random
from collections import OrderedDict

import numpy as np

from posec_core import cputime

logger = logging.getLogger(__name__)

class PlayCounts(object):

    def __init__(self, agg):
        self.player_indices = OrderedDict()
        strategy_space_length = 0
        self.counts = []
        for player in agg.N:
            start_idx = strategy_space_length
            n_actions = len(agg.S[player])
            strategy_space_length += n_actions
            # Uniform prior
            self.counts.extend([1 / float(n_actions)] * n_actions)

            # Random pure prior
            # rand_choice = random.randrange(n_actions)
            # self.counts.extend([int(j == rand_choice) for j in range(n_actions)])

            self.player_indices[player] = (start_idx, strategy_space_length)

    def update_model(self, new_counts):
        self.counts = np.add(self.counts, new_counts)

    def get_player_strategy(self, player):
        """Normalize counts"""
        player_counts = self.counts[self.player_indices[player][0]:self.player_indices[player][1]]
        return normalize(np.array(player_counts))

    def get_mixed_strategy(self):
        s = []
        for i in self.player_indices.keys():
            s.extend(self.get_player_strategy(i))
        return s

def strategy_string(mixed_strategy):
    return ' '.join(map(str, mixed_strategy)) + '\n'

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    return v/norm

def compute_expected_utilities(s, i_start, i_end, agg, i):
    """Returns EU of playing each action"""
    n_actions = i_end - i_start
    expected_utilities = []
    for j in range(n_actions):
        # Set player i to play action j
        s[i_start:i_end] = [int(p == j) for p in range(n_actions)]
        # Calculate EU
        expected_utility = agg.test(strategy_string(s))[agg.N.index(i)]
        expected_utilities.append(expected_utility)
    return expected_utilities

def FP(agg, seed=None, output=None, truthful_init=True, initial_strategies=None, stop_cycle=True, cutoff=3600):
    start = cputime()

    if seed is not None:
        random.seed(seed)

    file = output if output is not None else os.devnull

    with open(file, 'a') as f:
        f.write('max_regret,cputime' + '\n')

        # Set up every player's model of other players (allows for players to have different priors from each other)
        player_to_model = {i : PlayCounts(agg) for i in agg.N}

        updates = player_to_model[agg.N[0]].get_mixed_strategy()

        iteration = 0
        while not agg.isNE(strategy_string(updates)) and cputime() - start < cutoff:
            if iteration % 100 == 0:
                logging.info("Iteration %d" % iteration)
            # For each player, update the model through a fictitious play round
            updates = []
            for i in agg.N:
                # Get everyone else's mixed strategy
                s = player_to_model[i].get_mixed_strategy()
                i_start, i_end = player_to_model[i].player_indices[i][0], player_to_model[i].player_indices[i][1]
                expected_utilities = compute_expected_utilities(s, i_start, i_end, agg, i)

                # Get the indices of the best actions
                m = max(expected_utilities)
                updates.extend(normalize(np.array([int(v == m) for v in expected_utilities])))

            # if sum(updates) != len(agg.N):
            #     logger.warn("Updates sum to %.2f but only %d players..." % (sum(updates), len(agg.N)))
            #     raise
            # Regret calculations
            max_regret = agg.max_regret(strategy_string(updates))
            f.write(str(max_regret)+','+str(cputime()-start)+'\n')
            logger.info( "Max regret overall is %s" % (max_regret))
            iteration += 1

            # Everyone updates their counts together
            for i in agg.N:
                player_to_model[i].update_model(updates)

        # Either cutoff or NE
        if agg.isNE(strategy_string(updates)):
            logger.info("NE FOUND at iteration %d" % iteration)
            # print agg.interpretProfile(strategy_string(updates))
        else:
            logger.info("Cutoff reached")

if __name__ == '__main__':
    from pyagg import *
    logging.basicConfig(format='%(asctime)-15s [%(levelname)s] %(message)s', level=logging.INFO, filename='posec.log')
    logging.getLogger().addHandler(logging.StreamHandler())
    agg = AGG_File('/ubc/cs/research/arrow/newmanne/positronic-economist/baggs/GFP/gfp_10_10_1_WBSI.bagg')
    FP(agg)