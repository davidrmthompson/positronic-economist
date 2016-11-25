from experiments import cputime
import random
import os
import logging
from posec_core import ProjectedMechanism
import sys

logger = logging.getLogger(__name__)

# Introduce class so we can add attributes to dicts
class MyDict(dict):
    pass

def actions_dict(setting, m):
    actions = {}
    for i in range(setting.n):
        actions[i] = m.A(setting, i, setting.Theta[i])
    return actions

def make_random_pure_strategies(setting, actions):
    s = MyDict()
    for i in range(setting.n):
        s.agents = range(setting.n)
        s[i] = random.choice(actions[i])
    return s

def copy_strategies(strategies):
    strategies_tmp = MyDict()
    strategies_tmp.agents = strategies.agents
    for j in strategies_tmp.agents:
        strategies_tmp[j] = strategies[j]
    return strategies_tmp

def compute_utility(setting, m, i, strategies):
    if isinstance(m, ProjectedMechanism):
        po = m.M(setting, i, setting.Theta[i], strategies)
    else:
        po = m.M(setting, strategies)
    return sum([setting.u(i, setting.Theta, o, strategies[i]) * p for o, p in po])

def IBR(setting, m, seed=None, output=None, stop_cycle=True,cutoff=3600):

    start = cputime()

    if seed is not None:
        random.seed(seed)

    file = output if output is not None else os.devnull

    with open(file, 'a') as f:

        f.write('max_regret,cputime'+'\n')

        actions = actions_dict(setting, m)
        strategies = make_random_pure_strategies(setting, actions)

        # Calculate regret and best responses
        max_regret = sys.maxint

        if stop_cycle:
            seen = set()
            seen.add(tuple(strategies[k] for k in range(setting.n)))

        while max_regret != 0:
            if cputime() - start > cutoff:
                print "OUT OF TIME"
                break

            max_regrets = []
            player_to_action_to_regret = {}
            for i in range(setting.n):

                u = compute_utility(setting, m, i, strategies)
                player_to_action_to_regret[i] = MyDict()

                strategies_tmp = copy_strategies(strategies)

                for possible_action in actions[i]:
                    strategies_tmp[i] = possible_action
                    other_u = compute_utility(setting, m, i, strategies_tmp)
                    player_to_action_to_regret[i][possible_action] = other_u - u

                max_regrets.append(max(player_to_action_to_regret[i].values()))
                # logging.info("Max regret for player %d is %s" % (i, maxRegrets[i]))

            max_regret = max(max_regrets)
            logging.info( "Max regret overall is %s" % (max_regret))
            f.write(str(max_regret)+','+str(cputime()-start)+'\n')

            # Everyone does their best response
            for i in range(setting.n):
                strategies[i] = max(player_to_action_to_regret[i].iteritems(), key=lambda item: item[1])[0]

            # Check for loops
            if stop_cycle:
                identifier = tuple(strategies[k] for k in range(setting.n))
                if identifier in seen:
                    logger.info("LOOP DETECTED, randomizing strategies")
                    strategies = make_random_pure_strategies(setting, actions)
                else:
                    seen.add(identifier)

        if max_regret == 0:
            print "Converged to equilibrium! Yay!"