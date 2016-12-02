The Positronic Economist
====================

A computational system for analyzing economic mechanisms

To solve games with additional solvers, download the AGG solvers at [the AGG website](http://agg.cs.ubc.ca/)

Also see the [PosEc website](https://www.cs.ubc.ca/research/posec/) for links to publications.

Questions, bug reports and feature suggestions should be directed to Neil Newman - newmanne at cs dot ubc dot ca

## Authors & Collaborators
PosEc is the product of the ideas and hard work of David Thompson, Neil Newman, and Kevin Leyton-Brown.

## Example
Here is the code needed to define a plurality voting game. 

```python
import itertools
from posec import *

#
# Setting
#

# 10 agents
n = 10 
# 3 outcomes
O = ("A", "B", "C") 
# Every preference ordering is possible
Theta = [list(itertools.permutations(O))] 
# Every agent is equally likely to have every preference ordering
P = [UniformDistribution(Theta)] * n 

# Utility function - 0 points if least preferred candidate wins, 1 if next most preferred etc.
def u(i, theta, o, a_i):
  return theta[i].index(o)
  
# Create the setting object
s = BayesianSetting(n, O, Theta, P, u)

#
# Mechanism
#

# Action function: the possible actions for each player (here to vote for any candidate)
def A(setting, i, theta_i):
  return setting.O

# The choice function - given all of the agents' actions, return a distribution over outcomes
def M(setting, a_N):
  scores = {o: a_N.count(o) for o in setting.O}
  maxScore = max(scores.values())
  winners = [o for o in scores.keys() if scores[o] == maxScore]
  return UniformDistribution(winners)

# Create the mechanism object
m = Mechanism(A, M)

# Run our structure inference algorithms to make a BAGG
agg = makeAGG(s, m, symmetry=True)

# Save it to a file for use with any tool that supports BAGGs!
agg.saveToFile('plurality.bagg')
```

Be sure to check out [our examples section](posec/applications/) that defines mechanisms and settings for position auctions and voting games. You are encouraged to contribute further mechanisms and settings!

## Release Notes

### PosEc 1.0 [1/12/2016]

* First official release of **PosEc**.
