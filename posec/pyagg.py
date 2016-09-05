''' Python wrapper class for representing action-graph games '''
import os
import re
import string

# Configuration options
# FIXME: this shouldn't be called solvers, but rather binaries, b/c I use
# getpayoffs and bgetpayoffs.  Also, it should probably live elsewhere.
AGG_SOLVER_PATH = ""


def _delta(n, k):
    return [0.0] * k + [1.0] + [0.0] * (n - k - 1)

" Goal: This should work with any AGG, not just mine. "


class _UniqueString(str):

    ''' Utility class for working with AGG_Files; using this class it's possible
to distinguish when two references point to the same agent, as opposed to two
different agents with the same name (ditto for actions and types) '''

    def __eq__(self, other):
        return id(self) == id(other)

FN_TYPE_SUM = 0
FN_TYPE_OR = 1
FN_TYPE_WEIGHTED_SUM = 10
FN_TYPE_WEIGHTED_MAX = 12


class AGG_File:

    def __init__(self, filename):
        self.filename = filename
        self.data = open(filename, "r").read()
        if self.data[:5] != "#AGG\n":
            print "WARNING: File does not have the #AGG tag: " + filename
        self._loadDetails()
        del self.data  # No need to keep that information hanging around
        # These will be popen streams
        self.tochild, self.fromchild = None, None

    def _getObjectNames(self, objectType, objectCount=None):
        """ Find all the names for a set of objects based on comments
            e.g., agents, actions, types """
        data = {}
        for line in re.findall("# %s [0-9]+:.*?\n" % (objectType), self.data):
            num = int(re.findall("[0-9]+", line)[0])
            assert num >= 0
            name = string.strip(string.split(line, ":", 1)[1])
            data[num] = _UniqueString(name)
        # print data
        if objectCount == None:
            objectCount = max(data.keys()) + 1
        output = ["%s%d" % (objectType, i) for i in range(objectCount)]
        for i in range(objectCount):
            if data.has_key(i):
                output[i] = data[i]
        return output

    def __getNames(self, n, m):
        self.N = self._getObjectNames("Agent", n)
        self.A = self._getObjectNames("Node", m)

    def _loadDetails(self):
        body = re.sub("#.*?\n", "", self.data)
        values = string.split(body)
        n = int(values[0])
        m = int(values[1])
        self.__getNames(n, m)
        aSizes = map(int, values[3:n + 3])
        index = n + 3
        actionIndexSets = []
        self.S = {}
        for i in range(n):
            agent = self.N[i]
            k = aSizes[i]
            self.S[agent] = [self.A[int(x)] for x in values[index:index + k]]
            index += k
        self.aSizes = aSizes

    def _popen(self):
        if self.tochild == None or self.fromchild == None:
            (self.tochild, self.fromchild) = os.popen2(
                AGG_SOLVER_PATH + "getpayoffs " + self.filename)

    def parse(self, strategyString):
        strategyString = self.fixStrategy(strategyString)
        output = []
        nPreviousActions = 0
        for i in range(len(self.aSizes)):
            nA = self.aSizes[i]
            strategy = string.split(strategyString)[
                nPreviousActions:nPreviousActions + nA]
            output.append(map(float, strategy))
            nPreviousActions += nA
        return output

    def interpretProfile(self, sp):
        sp = string.split(self.fixStrategy(sp))
        output = []
        for i in self.N:
            aSize = len(self.S[i])
            strat = sp[:aSize]
            if set(strat) != set(["0", "1"]):
                output.append("MIXED")
            else:
                sp = sp[aSize:]
                action = strat.index("1")
                output.append(self.S[i][action])
        return output

    def fixStrategy(self, stratStr):
        stratStr = string.strip(stratStr)
        if stratStr[:3] == "NE,":
            stratStr = stratStr[3:]
        strat = string.split(stratStr, ",")
        if len(strat) == 1:
            strat = string.split(stratStr)
        strat2 = []
        for item in strat:
            try:
                top, bottom = string.split(item, "/")
                strat2.append(str(float(top) / float(bottom)))
            except:
                strat2.append(item)
        return string.join(strat2)

    def test(self, strategyString):
        ''' Returns a n-length vector of the agents' payoffs '''
        self._popen()
        strategyString = self.fixStrategy(strategyString)
        # print strategyString
        self.tochild.write(strategyString + "\n")
        self.tochild.flush()
        self.fromchild.readline()
        output = map(float, string.split(
            string.strip(self.fromchild.readline())))
        return output

    def isNE(self, strategyString):
        ''' Tests whether or not a given strategy profile is a (Bayes) Nash equilibrium '''
        strategyString = self.fixStrategy(strategyString)
        rc = True
        eu = self.test(strategyString)
        LL = fromLtoLL(map(float, string.split(strategyString)), self.aSizes)
        LL2 = fromLtoLL(map(float, string.split(strategyString)), self.aSizes)
        for i in range(len(self.aSizes)):
            nA = len(LL[i])
            for a in range(nA):
                LL2[i] = _delta(nA, a)
                payoff = self.test(fromLLtoString(LL2))[i]
                # print i,a,payoff-eu[i],"*"*(LL[i][a]>0.0)
                if payoff > eu[i]:
                    rc = False
                    # Comment out to test for the regret of every action
                    return rc
            LL2[i] = LL[i]
        return rc

    def __del__(self):
        try:
            self.tochild.close()
        except:
            pass
        try:
            self.fromchild.close()
        except:
            pass


class BAGG_File(AGG_File):

    """
    >>> import AGG_Examples
    >>> bagg = AGG_Examples.MixedValueChicken()
    >>> bagg.saveToFile("mvc.bagg")
    >>> bagg.isNE("1 0 1 0 1 0 1 0")
    False
    >>> bagg.testExAnte("1 0 1 0 1 0 1 0")
    [0.0, 0.0]
    >>> baggf = BAGG_File("mvc.bagg")
    >>> baggf.Theta
    ['High', 'Low']
    >>> baggf.S["High"]
    ["('High', 'Swerve')", "('High', 'Straight')"]
    >>> baggf.P
    {'1': {'High': 0.5, 'Low': 0.5}, '2': {'High': 0.5, 'Low': 0.5}}
    >>> baggf.isNE("1 0 1 0 1 0 1 0")
    False
    >>> baggf.testExAnte("1 0 1 0 1 0 1 0")
    [0.0, 0.0]
"""

    def _getNames(self, n, m):
        self.N = self._getObjectNames("Agent", n)
        self.A = self._getObjectNames("Node", m)
        self.Theta = self._getObjectNames("Type")

    def _loadDetails(self):
        body = re.sub("#.*?\n", "", self.data)
        self.values = string.split(body)
        self.offset = 0

        def nextValues(n, form=int):
            end = self.offset + n
            out = map(form, self.values[self.offset:end])
            self.offset = end
            if n == 1:
                return out[0]
            return out
        n = nextValues(1)
        m = nextValues(1)
        self._getNames(n, m)
        nextValues(1)
        typeCounts = nextValues(n)
        pFloats = nextValues(sum(typeCounts), float)
        aSizes = nextValues(sum(typeCounts))
        self.aSizes = aSizes

        # FIXME: this will currently choke if the types aren't stored properly!
        nextType = 0
        self.S = {}
        self.P = {}
        agentTypeOffset = 0
        for i in range(n):
            self.P[self.N[i]] = {}
            for j in range(typeCounts[i]):
                actions = nextValues(aSizes[agentTypeOffset])
                actions = [self.A[k] for k in actions]
                if actions in self.S.values():
                    t = [t for t in self.S.keys() if self.S[t] == actions][0]
                else:
                    t = self.Theta[nextType]
                    self.S[t] = actions
                    nextType += 1
                self.P[self.N[i]][t] = pFloats[agentTypeOffset]
                agentTypeOffset += 1

        # FIXME: What about two types that had exactly the same set of action nodes?
        # Answer, just have a meltdown!  Maybe build that into the type output
        # on BAGG

    def test(self, strategyString):
        ''' Returns an ex interim expected payoff profile '''
        self._popen()
        strategyString = self.fixStrategy(strategyString)
        # print strategyString
        self.tochild.write(strategyString + "\n")
        self.tochild.flush()
        self.fromchild.readline()
        output = map(float, string.split(
            string.strip(self.fromchild.readline())))
        self.fromchild.readline()
        self.fromchild.readline()
        return output

    def testExAnte(self, strategyString):
        self._popen()
        strategyString = self.fixStrategy(strategyString)
        # print strategyString
        self.tochild.write(strategyString + "\n")
        self.tochild.flush()
        self.fromchild.readline()
        self.fromchild.readline()
        self.fromchild.readline()
        output = map(float, string.split(
            string.strip(self.fromchild.readline())))
        return output

    def _popen(self):
        if self.tochild == None or self.fromchild == None:
            (self.tochild, self.fromchild) = os.popen2(
                AGG_SOLVER_PATH + "bgetpayoffs " + self.filename)


class AGG(AGG_File):

    def sizeAsAGG(self):
        return sum([len(self._possibleConfigs(a)) for a in self.A])

    def sizeAsNFG(self):
        size = 1L
        for i in self.N:
            size *= len(self.S[i])
        return len(self.N) * size

    def __init__(self, N, A, S, F, v, f, u, title=None):
        ''' N is a list of players
            A is a list of actions
            S is a mapping from players to lists of actions
            F is a list of projection (aka function) nodes
            v is a list of arcs (2-tupes of start,end nodes)
            f is a mapping from projection node to type of projection (integer)
            u is a mapping from an action node to a payoff mapping (tuples of inputs to real values)

    >>> import AGG_Examples
    >>> agg = AGG_Examples.PrisonersDilemma()
    >>> agg.saveToFile("pd.agg")
    >>> agg.test("NE,1,0,1,0")
    [3.0, 3.0]
    >>> agg.isNE("NE,1,0,1,0")
    False
    >>> string.strip(simpdiv.solve(agg).next())
    'NE,0,1,0,1'
    >>> string.strip(gnm.solve(agg).next())
    '0      1      0      1'
    >>> f = open("pd2.agg","w")
    >>> f.write(open("pd.agg","r").read()[5:])
    >>> f.close()
    >>> aggf = AGG_File("pd2.agg")
    WARNING: File does not have the #AGG tag: pd2.agg
    >>> aggf.test("NE,1,0,1,0")
    [3.0, 3.0]
    >>> aggf.isNE("NE,1,0,1,0")
    False
    >>> string.strip(sem.solve(aggf).next())
    'NE,0,1,0,1'
            '''
        self.N = N
        self.A = A
        self.S = S
        self.F = F
        self.v = v
        self.f = f
        self.u = u
        self.title = title
        self._testForExceptions()
        self.baseName = ".tmp"
        self._makeArcDict()
        self._makeEffectDict()
        self._hasCascadingFNsCache = None
        # These will be popen streams
        self.tochild, self.fromchild = None, None
        self.filename = None
        self.aSizes = map(len, self.S.values())

    def arcsTo(self, node):
        return [arc for arc in self.v if arc[1] == node]

    def neighbours(self, node):
        return [arc[0] for arc in self.v if arc[1] == node]

    def _makeArcDict(self):
        self.arcDict = {}
        for arc in self.v:
            key = (arc[:2])
            self.arcDict[key] = arc
        self.inArcDict = {}
        for arc in self.v:
            key = (arc[1])
            if not self.inArcDict.has_key(key):
                self.inArcDict[key] = []
            self.inArcDict[key].append(arc)

    def _makeEffectDict(self):
        self.effectDict = {}
        for act1 in self.A:
            for act2 in self.A:
                if act1 == act2:
                    self.effectDict[(act1, act2)] = 1
                else:
                    self.effectDict[(act1, act2)] = 0
        for act in self.A:
            for node in self.F:
                arc = self._getArc(act, node)
                if arc == None:
                    self.effectDict[(act, node)] = 0
                    continue
                fType = self.f[node]
                if fType == FN_TYPE_SUM:
                    self.effectDict[(act, node)] = 1
                    continue
                if fType == FN_TYPE_OR:
                    self.effectDict[(act, node)] = "*"
                    continue
                if fType[0] == FN_TYPE_WEIGHTED_SUM:
                    self.effectDict[(act, node)] = arc[2]
                    continue
                if fType[0] == FN_TYPE_WEIGHTED_MAX:
                    self.effectDict[(act, node)] = "*"
                    continue

    def _testForExceptions(self):
        return  # This is sucking up a lot of time
        ''' Makes sure that the AGG is self-consistent '''
        A2 = set()
        for actions in self.S.values():
            A2 = A2.union(actions)
        for a in self.A:
            if a not in A2:
                raise "No agent can perform action " + str(a)

        for i in self.N:
            if not self.S.has_key(i):
                raise "No action set for player " + str(i)
        for i in self.S.keys():
            if i not in self.N:
                raise "Action set for non-existant player " + str(i)
            for a in self.S[i]:
                if a not in self.A:
                    raise "Action " + \
                        str(a) + " is not in A, but is in the action set of Player " + str(
                            i)
        Nodes = self.A + self.F
        for arc in self.v:
            start, end = arc[0:2]
            if start not in Nodes:
                raise "Arc from non-existant node " + str(start)
            if end not in Nodes:
                raise "Arc to non-existant node " + str(end)

    def _saveActionSets(self, file):  # Gets overloaded by BAGGs
        ''' then we have n numbers specifying |A| for each player '''
        file.write("# |Action sets|\n")
        for i in self.N:
            file.write(str(len(self.S[i])) + " ")
        ''' then we have the action indices for each player i'''
        file.write("\n\n# Action sets\n")
        for i in self.N:
            for s in self.S[i]:
                file.write(str(self.A.index(s)) + " ")
            file.write("\n")

    def saveToFile(self, filename):
        self._makeArcDict()
        self._makeEffectDict()
        ''' Saves the game to a formal readible by Albert Xin Jiang's solvers '''
        Nodes = self.A + self.F
        self.filename = filename
        file = open(filename, 'w')
        file.write("#AGG\n")
        ''' the first thing we have is a null-terminated string that is the title'''
        if self.title:
            file.write("# %s\n\n" % (self.title))
        ''' then we have 3 numbers: n, A, F'''
        file.write(
            "# Players\n%d\n# Action Nodes\n%d\n# Function nodes\n%d\n" %
            (len(self.N), len(self.A), len(self.F)))
        for i, name in zip(range(len(self.N)), self.N):
            file.write("# Agent %d:%s\n" % (i, str(name)))

        # Gets overloaded for BAGGs
        self._saveActionSets(file)
        ''' then we have, for each Node, the source of each arc '''
        file.write("\n\n# Arcs\n")
        for i in range(len(Nodes)):
            n = Nodes[i]
            file.write("# Node %d: %s\n" % (i, str(n)))
            neighbours = self.neighbours(n)
            file.write(str(len(neighbours)) + " ")
            for n2 in neighbours:
                file.write(str(Nodes.index(n2)) + " ")
##                file.write(str(Nodes.index(arc[0])) + " ")
# try:
##                arcs = self.inArcDict[(n)]
# except:
##                arcs = []
# arcs = [(n2,None) for n2 in self.A + self.F if self._hasArc(n2, n)] # Hack experiment
##            file.write(str(len(arcs)) + " ")
# for arc in arcs:
##                file.write(str(Nodes.index(arc[0])) + " ")
            file.write("\n")
        ''' then we have P numbers, each representing the type of each function node '''
        file.write("\n\n# Types of function nodes\n")
        for fnNode in self.F:
            fnNodeSig = self.f[fnNode]
            if type(fnNodeSig) == type(1):  # FIXME: bad style!
                file.write(str(fnNodeSig) + "\n")
            else:
                # The function type and the default value
                assert len(fnNodeSig) == 2
                file.write("%d %d [" % (fnNodeSig))
                for a in self.A:
                    key = ((a, fnNode))
                    if self.arcDict.has_key(key):
                        file.write(str(self.arcDict[key][2]) + " ")
                    else:
                        file.write("0 ")
                file.write("]\n")
        ''' now for each action node, have
        name (null terminated string),
        the number 1
        some utility value of some sort
        string that represents the key 
        utility
        '''

        file.write("\n\n# Payoffs of action nodes\n")
        for a in self.A:  # a is a string typically
            file.write("# Action node: %s\n" % (str(a)))
            self._savePayoffs(a, file)
        file.close()

    def _savePayoffs(self, a, file):
        t = type(self.u[a])
        if t == type({}):
            return self._savePayoffsDict(a, file)
        if "__call__" in dir(self.u[a]):
            return self._savePayoffsFunc(a, file)
        raise "Invalid format for payoffs of action:" + str(a)

    def _savePayoffsDict(self, a, file):
        file.write("1 " + str(len(self.u[a].keys())) + "\n")  # how many keys
        keys = self.u[a].keys()
        keys.sort()  # For neatness
        for key in keys:
            try:
                keyStr = string.join(map(str, key))
            except Exception:
                keyStr = str(key)
            file.write("  [ " + keyStr + " ] " + str(self.u[a][key]) + "\n")

    def _savePayoffsFunc(self, a, file):
        assert not self._hasCascadingFNs()
        configs = self._possibleConfigs(a)
        file.write("1 " + str(len(configs)) + "\n")  # how many keys
        configs.sort()  # For neatness
        for config in configs:
            keyStr = string.join(map(str, config))
            file.write("  [ " + keyStr + " ] " + str(self.u[a](config)) + "\n")

    def _getArc(self, start, end):
        try:
            return self.arcDict[((start, end))]
        except:
            return None

    def _hasArc(self, start, end):
        return self.arcDict.has_key(((start, end)))

    def _updateConfigurationNode(agg, node, config, action):
        ''' Assumes no cascading of function nodes '''
        if not agg._hasArc(action, node):
            return config
        if node in agg.A:
            if action == node:
                return config + 1
            return config
        fnParams = agg.f[node]
        if fnParams == FN_TYPE_SUM:
            return config + 1
        if fnParams == FN_TYPE_OR:
            return min([config + 1, 1])
        if fnParams[0] == FN_TYPE_WEIGHTED_SUM:
            return config + agg._getArc(action, node)[2]
        if fnParams[0] == FN_TYPE_WEIGHTED_MAX:
            arcWeight = agg._getArc(action, node)[2]
            if arcWeight > config:
                return arcWeight
            return config
        raise "Only works for a specific set of function nodes, right now."

    def _quickUpdateConfigurationNode(agg, node, config, action):
        key = (action, node)
        effect = agg.effectDict[key]
        if effect == "*":
            return agg._updateConfigurationNode(node, config, action)
        return config + effect

    def _updateConfiguration(agg, nodes, config, action):
        return tuple([(agg._quickUpdateConfigurationNode(nodes[n], config[n], action)) for n in range(len(nodes))])

    def _getConfigurationsSlow(self, nodes, holdBackAgents=[]):
        Cs = [[0] * len(nodes)]
        for i in self.N:
            if i in holdBackAgents:
                continue
            C2 = []
            for a in self._possibleActions(i):
                for c in Cs:
                    newc = self._updateConfiguration(nodes, c, a)

                    if newc not in C2:
                        C2.append(newc)
            Cs = C2
        return Cs

    def _getConfigurations(self, nodes, holdBackAgents=[]):
        effectSets = []
        for i in self.N:
            if i in holdBackAgents:
                continue
            effectSets.append([])
            for a in self._possibleActions(i):
                effect = [(self.effectDict[(a, node)]) for node in nodes]
                if "*" in effect:
                    return self._getConfigurationsSlow(nodes, holdBackAgents)
                # Fall back to the old, slower method
                if effect not in effectSets[-1]:
                    effectSets[-1].append(effect)
        # print effectSets
        Cs = [[0] * len(nodes)]
        # A bit kludgey, but handles richer function nodes
        for i in range(len(nodes)):
            node = nodes[i]
            try:
                if agg.f[node][0] in [FN_TYPE_WEIGHTED_SUM, FN_TYPE_WEIGHTED_MAX]:
                    Cs[i] = agg.f[node][1]
            except:
                pass
        for effectSet in effectSets:
            C2 = []
            for effect in effectSet:
                for c in Cs:
                    newc = tuple([a + b for a, b in zip(effect, c)])
                    if newc not in C2:
                        C2.append(newc)
            # print Cs, effectSet, C2
            Cs = C2
        return Cs

    def _possibleActions(self, player):
        assert player in self.N
        return self.S[player]

    def _possibleConfigs(self, a):
        neighbours = self.neighbours(a)
        players = [i for i in self.N if a in self._possibleActions(i)]
        configs = []
        for i in players:
            configsWithI = self._getConfigurationsSlow(neighbours, [i])
            for c in configsWithI:
                c = self._updateConfiguration(neighbours, c, a)
                if c not in configs:
                    configs.append(c)
        return configs

    def _hasCascadingFNs(self):
        if self._hasCascadingFNsCache != None:
            return self._hasCascadingFNsCache
        for f in self.F:
            for f2 in self.F:
                if self._hasArc(f, f2):
                    self._hasCascadingFNsCache = True
                return self._hasCascadingFNsCache
        self._hasCascadingFNsCache = False
        return False

    def _symmetric(self, agent1, agent2):
        return set(self.S[agent1]) == set(self.S[agent2])

    def _agentPartition(self):
        partitions = []
        for agent in self.N:
            assigned = False
            for partition in partitions:
                if self._symmetric(agent, partition[0]):
                    partition.append(agent)
                    assigned = True
                    break
            if not assigned:
                partitions.append([agent])
        return partitions


class BAGG(BAGG_File, AGG):

    def sizeAsNFG(self):
        ''' Returns the size of the games as a Bayesian normal-form game '''
        size = 1L
        for i in self.N:
            typeActionPairs = 0L
            for theta in self.P[i].keys():
                if self.P[i][theta] > 0:
                    typeActionPairs += len(self.S[theta])
            size *= typeActionPairs
        return len(self.N) * size

    def __init__(self, N, Theta, P, A, S, F, v, f, u, title=None):
        ''' N is a list of players
            Theta is a list of types
            P is a mapping from players to mappings from types to probability
            A is a list of actions
            S is a mapping from types to lists of actions
            F is a list of function nodes
            v is a list of arcs (2-tupes of start,end nodes)
            f is a mapping from projection node to type of projection (integer)
            u is a mapping from an action node to a payoff mapping (tuples of inputs to real values)
            '''
        self.N = N
        self.Theta = Theta
        self.P = P
        self.A = A
        self.S = S
        self.F = F
        self.v = v
        self.f = f
        self.u = u
        self.title = title
        self._testForExceptions()
        self.baseName = ".tmp"
        self._makeArcDict()
        self._makeEffectDict()
        self._hasCascadingFNsCache = None
        self.filename = None
        # These will be popen streams
        self.tochild, self.fromchild = None, None
        self.aSizes = []
        for i in self.N:
            for t in self.Theta:
                if self.P[i][t] > 0:
                    self.aSizes.append(len(self.S[t]))

    def _testForExceptions(self):
        pass
##        ''' Makes sure that the AGG is self-consistent '''
# return None # Can't run on Python 2.3 because there's no set object
##        A2 = set()
# for actions in self.S.values():
##            A2 = A2.union(actions)
# for a in self.A:
# if a not in A2:
##                raise "No type can perform action "+str(a)
#
# for i in self.N:
# if not self.P.has_key(i):
##                raise "No action set for player "+str(i)
# for i in self.P.keys():
# if i not in self.N:
##                raise "Action set for non-existant player "+str(i)
##            pi = P[i]
# for theta in pi.keys():
##                assert theta in Theta
#
##        Nodes = self.A+self.F
# for start,end in self.v:
# if start not in Nodes:
##                raise "Arc from non-existant node "+str(start)
# if end not in Nodes:
##                raise "Arc to non-existant node "+str(end)

    def _saveActionSets(self, file):
        for i in range(len(self.Theta)):
            file.write("# Type %d: %s\n" % (i, self.Theta[i]))
        file.write("\n")
        for i in self.N:
            nonZeroPTypes = sum([(p > 0.0) for p in self.P[i].values()])

            file.write(str(nonZeroPTypes) + " ")
        file.write("\n")
        for i in self.N:
            for theta in self.Theta:
                if not self.P[i].has_key(theta):
                    continue
                if self.P[i][theta] <= 0.0:
                    continue
                file.write(str(self.P[i][theta]) + " ")
            file.write("\n")

        file.write("\n")

        for i in self.N:
            for theta in self.Theta:
                if not self.P[i].has_key(theta):
                    continue
                if self.P[i][theta] <= 0.0:
                    continue
                file.write(str(len(self.S[theta])) + " ")
            file.write("\n")
        file.write("\n")

        for i in self.N:
            for theta in self.Theta:
                if not self.P[i].has_key(theta):
                    continue
                if self.P[i][theta] <= 0.0:
                    continue
                for a in self.S[theta]:
                    file.write(str(self.A.index(a)) + " ")
                file.write("\n")

    def _canPlayAction(self, player, action):
        assert player in self.N
        assert action in self.A
        for theta in self.Theta:
            if action in self.S[theta] and self.P[player][theta] > 0.0:
                return True
        return False

    def _possibleActions(self, player):
        assert player in self.N
        return [a for a in self.A if self._canPlayAction(player, a)]

    def _symmetric(self, agent1, agent2):
        return set(self.P[agent1]) == set(self.P[agent2])


def fromLtoLL(L, aSizes):
    output = []
    ind = 0
    for size in aSizes:
        output.append(L[ind:ind + size])
        ind += size
    return output


def __isDelta(s):
    for e in s:
        if e not in [0.0, 1.0]:
            return False
    if sum(s) != 1.0:
        return False
    return True


def fromLLtoLC(LL):
    output = []
    for s in LL:
        if not __isDelta(s):
            output.append(s)
        else:
            output.append(s.index(1.0))
    return output


def fromLLtoString(LL, actionDelim=" ", agentDelim="\t"):
    return string.join([(string.join(map(str, s), actionDelim)) for s in LL], agentDelim)


class _Solver:

    def __init__(self, cmd, defaults={}):
        self.cmd = cmd
        self.defaults = defaults

    def solve(self, agg, timeLimit=None, **options):
        assert agg.filename != None
        command = self.cmd.format(
            filename=agg.filename, **dict(options.items() + self.defaults.items()))
        if timeLimit != None:
            command = "timerun " + str(timeLimit) + " " + command
        for line in os.popen(command, 'r').xreadlines():
            if line[:3] == "NE,":
                yield line


class _GNM(_Solver):
    # Needs a subclass because it doesn't output equilibria in the same way as
    # the others

    def solve(self, agg, timeLimit=None, **options):
        assert agg.filename != None
        command = self.cmd.format(
            filename=agg.filename, **dict(options.items() + self.defaults.items()))
        if timeLimit != None:
            command = "timerun " + str(timeLimit) + " " + command
        lastLine = None
        for line in os.popen(command, 'r').xreadlines():
            if line == "Expected utility for each player under a sample eqlm:\n":
                yield lastLine
            lastLine = line

sem = _Solver("sem_agg -uran-2 <{filename}")
gnm = _GNM("gnm_agg -f {filename} {seed} {runs}", {"seed": 1, "runs": 1})
simpdiv = _Solver(
    "makeStart {filename} {filename}.start {seed};simpdiv -s {filename}.start <{filename}", {"seed": 1})

SOLVERS = [sem, gnm, simpdiv]


def purgeBarren(agg):
    ''' Removes any functions nodes that have no children '''
    changed = True
    while changed:
        barren = []
        for f in agg.F:
            outputs = [e[1] for e in agg.v if e[0] == f]
            if not outputs:
                barren.append(f)
        changed = bool(barren)
        for f in barren:
            agg.F.remove(f)
            agg.v = filter(lambda x: x[1] != f, agg.v)
        agg._makeArcDict()
        agg._makeEffectDict()

if __name__ == "__main__":
    import doctest
    doctest.testmod()
