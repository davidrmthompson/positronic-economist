import pyagg
from pyagg import *
import posec


def preprocess(agg):
    for act in agg.A:
        configs = agg._possibleConfigs(act)
        if callable(agg.u[act]):
            mapping = dict([(c, agg.u[act](c)) for c in configs])
        else:
            mapping = dict([(c, agg.u[act][c]) for c in configs])
        agg.u[act] = mapping


def collapseTest(f, C, t):
    newF = {}
    for c in C:
        ct = t(c)
        fc = f[c]
        if ct in newF.keys():
            if newF[ct] != fc:
                return None  # Transform is inconsistent
        else:
            newF[ct] = fc
    # Additional bar: t makes the set of configurations smaller
    return newF


def testCut(agg, act, arcIndex):
    if not isinstance(arcIndex, int):
        arcIndex = findArcIndex(agg, act, arcIndex)
    ''' Returns a new utility mapping if the arc can be cut.
Returns None if not. '''
    def t(c):
        c = list(c)
        c = c[:arcIndex] + c[arcIndex + 1:]
        return tuple(c)
    return collapseTest(agg.u[act], agg.u[act].keys(), t)


def tryCutNode(agg, act, node):
    return tryCut(agg, act, findArcIndex(agg, act, node))


def tryCut(agg, act, arcIndex, strict=False):
    ''' If possible (and, optionally, strictly improving) cut an arc '''
    newMapping = testCut(agg, act, arcIndex)
    if newMapping == None:
        return False
    if len(newMapping) == len(agg.u[act]) and strict:
        return False
    inarcs = [arc for arc in agg.v if arc[1] == act]
    agg.v = [arc for arc in agg.v if arc[1] != act] + \
        inarcs[:arcIndex] + inarcs[arcIndex + 1:]
    agg.u[act] = newMapping
    return True


def findArcIndex(agg, act, otherNode):
    inputs = [arc[0] for arc in agg.v if arc[1] == act]
    if otherNode not in inputs:
        raise "ASDF"
    return inputs.index(otherNode)
# FIXME: Is arcIndex the right way to identify them?  Or do I want to look
# it up?


def makeSum(agg, act, existingSums, weights, functionName):
    ''' Create a new weighted sum node that combines a bunch of existing inputs '''
    assert len(weights) == len(existingSums)
    inputs = [arc[0] for arc in agg.v if arc[1] == act]
    inputsAndIndeces = zip(inputs, range(len(inputs)))
    for fn in existingSums:
        if not fn in inputs:
            raise "makeSum can only merge existing sum inputs"
    inputsAndIndeces = [(n, i)
                        for n, i in inputsAndIndeces if n in existingSums]
    fn = functionName
    agg.F.append(fn)
    agg.f[fn] = (pyagg.FN_TYPE_WEIGHTED_SUM, 0)

    # Add the arcs
    actionWeights = [0] * len(agg.A)
    for j in range(len(inputsAndIndeces)):
        n, i = inputsAndIndeces[j]
        # print n,i,weights[j]
        arcsToSum = [arc for arc in agg.v if arc[1] == n]
        for a, __, w in arcsToSum:
            actionWeights[agg.A.index(a)] += w * weights[j]
    # print actionWeights
    for a, i in zip(agg.A, range(len(agg.A))):
        if actionWeights[i]:
            agg.v.append((a, fn, actionWeights[i]))
            # print agg.v[-1]
    agg.v.append((fn, act))

    # New utility mapping
    indeces = [i for n, i in inputsAndIndeces]

    def t(c):
        c = list(c)
        newValue = sum([weights[i] * c[indeces[i]]
                       for i in range(len(indeces))])
        c = c + [newValue]
        return tuple(c)
    agg.u[act] = collapseTest(agg.u[act], agg.u[act].keys(), t)


def makeOr(agg, act, sumNode, functionName):
    ''' Create a new weighted sum node that combines a bunch of existing inputs '''
    inputs = [arc[0] for arc in agg.v if arc[1] == act]
    assert sumNode in inputs
    inputsAndIndeces = zip(inputs, range(len(inputs)))
    fn = functionName
    agg.F.append(fn)
    agg.f[fn] = (pyagg.FN_TYPE_WEIGHTED_MAX, 0)

    # Add the arcs
    arcsToSum = [arc for arc in agg.v if arc[1] == sumNode]
    for a, __, w in arcsToSum:
        agg.v.append((a, fn, 1))
    agg.v.append((fn, act))

    # New utility mapping
    index = inputs.index(sumNode)

    def t(c):
        c = list(c)
        newValue = 1 * (c[index] > 0)
        c = c + [newValue]
        return tuple(c)
    agg.u[act] = collapseTest(agg.u[act], agg.u[act].keys(), t)


def makeMax(agg, act, existingMaxes, weights, functionName):
    assert len(weights) == len(existingMaxes)
    inputs = [arc[0] for arc in agg.v if arc[1] == act]
    inputsAndIndeces = zip(inputs, range(len(inputs)))
    inputsAndIndeces = [(n, i)
                        for n, i in inputsAndIndeces if n in existingMaxes]
    for fn in existingMaxes:
        if not fn in inputs:
            raise "makeMax can only merge existing max inputs"
    fn = functionName
    agg.F.append(fn)
    agg.f[fn] = (pyagg.FN_TYPE_WEIGHTED_MAX, 0)

    # Add the arcs
    actionWeights = [0] * len(agg.A)
    for j in range(len(inputsAndIndeces)):
        n, i = inputsAndIndeces[j]
        arcsToMax = [arc for arc in agg.v if arc[1] == n]
        for a, __, w in arcsToMax:
            w = int(w * weights[j])
            actionWeights[agg.A.index(a)] = max(
                [actionWeights[agg.A.index(a)], w])
    for a, i in zip(agg.A, range(len(agg.A))):
        if actionWeights[i]:
            agg.v.append((a, fn, actionWeights[i]))
            # print agg.v[-1]
    agg.v.append((fn, act))

    # New utility mapping
    indeces = [i for n, i in inputsAndIndeces]

    def t(c):
        c = list(c)
        newValue = max([int(weights[i] * c[indeces[i]])
                       for i in range(len(indeces))])
        c = c + [newValue]
        return tuple(c)
    agg.u[act] = collapseTest(agg.u[act], agg.u[act].keys(), t)


SUM_LOG = []


def compressByILS(agg, seed=None):
    import random
    if seed != None:
        random.seed(seed)
    assert isinstance(agg, AGG)
    # assert not isinstance(agg,BAGG) # FIXME: This should work too!

    # Preprocess step: convert all utility functions to utility mappings, and
    #   make sure that the utility mapping's domain is exactly the set of possible
    #   configurations.
    for act in agg.A:
        configs = agg._possibleConfigs(act)
        if callable(agg.u[act]):
            mapping = dict([(c, agg.u[act](c)) for c in configs])
        else:
            mapping = dict([(c, agg.u[act][c]) for c in configs])
        agg.u[act] = mapping

    def collapseTest(f, C, t):
        newF = {}
        for c in C:
            ct = t(c)
            fc = f[c]
            if ct in newF.keys():
                if newF[ct] != fc:
                    return None  # Transform is inconsistent
            else:
                newF[ct] = fc
        # Additional bar: t makes the set of configurations smaller
        return newF

    def makeSnapshot(agg, act):
        inarcs = [arc for arc in agg.v if arc[1] == act]
        return inarcs, agg.u[act]

    def restoreSnapshot(agg, act, snapshot):
        inarcs, mapping = snapshot
        agg.v = [arc for arc in agg.v if arc[1] != act] + inarcs
        agg.u[act] = mapping

    def cost(mapping):
        if isinstance(mapping, tuple):
            mapping = mapping[1]
        return len(mapping)
        config = mapping.keys()[0]
        inarcs = len(config)
        configs = len(mapping)
        return inarcs + configs

    def iterativeFirstImprovement(agg, act, minimumCuts=0):
        while True:
            arcs = [arc for arc in agg.v if arc[1] == act]
            arcNumbers = range(len(arcs))
            random.shuffle(arcNumbers)
            success = False
            for i in arcNumbers:
                if tryCut(agg, act, i, minimumCuts <= 0):
                    success = True
                    minimumCuts -= 1
                    break
            if not success:
                return

    def match(fn1, fn2):
        matches = 0
        if set(fn1.agents) == set(fn2.agents):
            matches += 1
        if set(fn1.types) == set(fn2.types):
            matches += 1
        if set(fn1.actions) == set(fn2.actions):
            matches += 1
        return matches

    def merge(fn1, fn2):
        function = fn1.function
        agents = tuple(set(fn1.agents).union(set(fn2.agents)))
        types = tuple(set(fn1.types).union(set(fn2.types)))
        actions = tuple(set(fn1.actions).union(set(fn2.actions)))
        actionWeights = []
        for a in actions:
            aw = 0
            if a in fn1.actions and fn1.action_weights != None:
                i = fn1.actions.index(a)
                aw += fn1.action_weights[i]
            if a in fn2.actions and fn2.action_weights != None:
                i = fn2.actions.index(a)
                aw += fn2.action_weights[i]
            actionWeights.append(aw)
        # FIXME: What about type weights
        return posec._PosEcAccessor(function, tuple(agents), tuple(types), None, actions, tuple(actionWeights), random.random())

    def makeOrName(fn1):
        tw = None
        aw = None
        if fn1.type_weights != None:
            tw = tuple([1 * (fn1.type_weights[i] > 0)
                       for i in range(len(fn1.type_weights))])
        if fn1.action_weights != None:
            aw = tuple([1 * (fn1.action_weights[i] > 0)
                       for i in range(len(fn1.action_weights))])
        return posec._PosEcAccessor("MAX", fn1.agents, fn1.types, tw, fn1.actions, aw, random.random())

    def randomIntWeight():
        i = 1
        prContinue = 0.1
        while True:
            if random.random() >= prContinue:
                return i
            i += 1

    def perturbBySum(agg, act):
        ''' Pick a bunch of sum nodes that feed into act and create a new sum node that combines
         them '''
        neighbours = [arc[0]
                      for arc in agg.v if arc[1] == act and arc[0].function == "SUM"]
        random.shuffle(neighbours)
        n1 = neighbours[0]
        success = False
        for n2 in neighbours[1:]:
            if match(n1, n2) == 2:  # FIXME This is rough!
                success = True
                break
        if not success:
            return False

        # Add the node
        fn = merge(n1, n2)
        makeSum(agg, act, [n1, n2], [randomIntWeight(), randomIntWeight()], fn)
        if testCut(agg, act, n1) != None and testCut(agg, act, n1) != None:
            tryCutNode(agg, act, n1)
            tryCutNode(agg, act, n2)
            return True
        else:
            tryCutNode(agg, act, fn)
            return False

    def perturbByOr(agg, act):
        neighbours = [arc[0]
                      for arc in agg.v if arc[1] == act and arc[0].function == "SUM"]
        n = random.choice(neighbours)
        fn = makeOrName(n)
        makeOr(agg, act, n, fn)
        if testCut(agg, act, n) != None:
            tryCutNode(agg, act, n)
            return True
        else:
            tryCutNode(agg, act, fn)
            return False

    def perturbByMax(agg, act):  # FIXME: Incomplete
        ''' Pick a bunch of sum nodes that feed into act and create a new sum node that combines
         them '''
        neighbours = [arc[0]
                      for arc in agg.v if arc[1] == act and agg.f[arc[0]] == (pyagg.FN_TYPE_WEIGHTED_MAX, 0)]
        random.shuffle(neighbours)
        n1 = neighbours[0]
        success = False
        for n2 in neighbours[1:]:
            if match(n1, n2) == 2:  # FIXME This is rough!
                success = True
                break
        if not success:
            return False

        # Add the node
        w = [random.expovariate(0.5), random.expovariate(0.5)]
        fn = merge(n1, n2)
        makeMax(agg, act, [n1, n2], w, fn)
        if testCut(agg, act, n1) != None and testCut(agg, act, n1) != None:
            tryCutNode(agg, act, n1)
            tryCutNode(agg, act, n2)
            return True
        else:
            tryCutNode(agg, act, fn)
            return False

    def perturb(agg, act):
        inputs = [arc[0] for arc in agg.v if arc[1] == act]
        nSums = len(
            [i for i in inputs if agg.f[i] == (pyagg.FN_TYPE_WEIGHTED_SUM, 0)])
        nMaxes = len(
            [i for i in inputs if agg.f[i] == (pyagg.FN_TYPE_WEIGHTED_MAX, 0)])
        moves = []
        if nSums >= 2:
            moves += [perturbBySum]
        if nSums >= 1:
            moves += [perturbByOr]
        if nMaxes >= 2:
            moves += [perturbByMax]
        # print "perturb",moves
        if not moves:
            return
        fn = random.choice(moves)
        fn(agg, act)

    c = 200
    s = 1
    restartP = 0.1
    for a in agg.A:
        original = makeSnapshot(agg, a)
        best = original
        print a, cost(agg.u[a]),
        iterativeFirstImprovement(agg, a, 999)
        c = cost(agg.u[a])
        print cost(agg.u[a]),
        import math
        c *= int(math.log(c, 2))
        # c =
        t = 0
        for i in range(c):
            for i in range(s):
                perturb(agg, a)
            iterativeFirstImprovement(agg, a, 999)
            if cost(agg.u[a]) < cost(best):
                best = makeSnapshot(agg, a)

            if random.random() < restartP:
                restoreSnapshot(agg, a, original)
            # posec.explain(agg,[a])
        restoreSnapshot(agg, a, best)
        iterativeFirstImprovement(agg, a, cost(agg.u[a]))
        print cost(agg.u[a])
        pyagg.purgeBarren(agg)
    agg._makeArcDict()
    agg._makeEffectDict()

# The tomfoolery around truthfulness in voting is a serious problem that I need to think hard about.
# The output its producing has some serious bullshit that I don't want a
# casual user to get tripped up by.


# FIXME: Too many calls to pre-process are happening.  That's not actually
# a cheap thing.
def anonymityCuts(agg):
    ''' pass '''
    preprocess(agg)
    for act in agg.A:
        # print act
        while True:
            arcsToAct = [arc for arc in agg.v if arc[1] == act]
            madeCut = False
            for arc, i in zip(arcsToAct, range(len(arcsToAct))):
                agents = []
                fnNode = arc[0]
                actsToFn = [arc[0] for arc in agg.v if arc[1] == fnNode]
                for act2 in actsToFn:
                    if act2.agent not in agents:
                        agents.append(act2.agent)
                if len(agents) != len(agg.N) and tryCut(agg, act, i, False):
                    madeCut = True
                    break
            if not madeCut:
                break
