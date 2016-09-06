from collections import namedtuple
import pyagg
import bbsi
import sys
import inspect
import time
import logging

def _hasAttribute(o, attrName):
    try:
        eval("o." + attrName)
        return True
    except AttributeError:
        return False


def _nArgs(fn):
    if not callable(fn):
        return -1
    return len(inspect.getargspec(fn).args)


class _PosEcKeyError(KeyError):
    pass


class _PosEcInternalException(Exception):
    pass


class _PosEcInputException(Exception):
    pass

# FIXME: If this was a class instead of a tuple, it could be a bit better
# (e.g., built in explain and merge commands)
_PosEcAccessor = namedtuple(
    "_PosEcAccessor", ["function", "agents", "types", "type_weights", "actions", "action_weights", "signature"])


class _InstrumentedDataStore(object):

    ''' Parent class for action and type profiles. '''

    def __init__(self, agg, a, data):
        ''' WARNING: Users of PosEc should not need to create instances of this class. '''
        self._agg = agg
        self._a = a
        self._data = data

    def _access(self, key):
        try:
            return self._data[key]
        except KeyError:
            raise _PosEcKeyError(key)

    # FIXME: A nice, but expensive sanity check
    def _symmetric(self):
        if all([a.agent is None for a in self._agg.A]):
            return True
        if all([a.agent is not None for a in self._agg.A]):
            return False
        raise _PosEcInternalException(
            "Action nodes are not consistently labeled with agent ids.")

    def _selectNodes(self, actions=None, agents=None, types=None):
        nodes = [n for n in self._agg.A]
        if actions is not None:
            for a in actions:
                assert a in self.actions
            nodes = [n for n in nodes if n.action in actions]
        if agents is not None:
            if self._a.agent is None:
                raise _PosEcInputException(
                    "agent-specific queries cannot be made when making a symmetric game.")
            for i in agents:
                assert i in self.agents
            nodes = [n for n in nodes if n.agent in agents]
        if types is not None:
            for theta in types:
                assert theta in self.types
            nodes = [n for n in nodes if n.type in types]
        return tuple(nodes)

    def _sum(self, **selection):
        assert "types" not in selection.keys() or "actions" not in selection.keys()
        if not "types" in selection.keys():
            selection["types"] = self.types
        if not "actions" in selection.keys():
            selection["actions"] = self.actions
        if not "agents" in selection.keys():
            selection["agents"] = self.agents
        if not selection["types"] or not selection["agents"] or not selection["actions"]:
            return 0
        return self._access(_PosEcAccessor("SUM", tuple(selection["agents"]), tuple(selection["types"]), None, tuple(selection["actions"]), tuple([1] * len(selection["actions"])), None))

    def _any(self, **selection):
        assert "types" not in selection.keys() or "actions" not in selection.keys()
        if not "types" in selection.keys():
            selection["types"] = self.types
            type_weights = None
        else:
            type_weights = tuple([1] * len(selection["types"]))
        if not "actions" in selection.keys():
            selection["actions"] = self.actions
            action_weights = None
        else:
            action_weights = tuple([1] * len(selection["actions"]))
        if not "agents" in selection.keys():
            selection["agents"] = self.agents
        if not selection["types"] or not selection["agents"] or not selection["actions"]:
            return False
        return self._access(_PosEcAccessor("MAX", tuple(selection["agents"]), tuple(selection["types"]), type_weights, tuple(selection["actions"]), action_weights, None)) == 1


class ActionProfile(_InstrumentedDataStore):

    ''' Representation of an action profile with efficient ways of accessing the data.

An ActionProfile is passed to a Mechanism's outcome function as the argument a_N
'''

    def count(self, a):
        ''' Returns a count of how many agents played action a '''
        return self.sum([a])

    def sum(self, A):
        ''' Returns a count of how many agents played any action in collection A '''
        if len(A) != len(set(A)):
            # FIXME: Maybe just a warning
            raise _PosEcInputException(
                "sum input A must not contain duplicate actions.")
        return self._sum(actions=A)

    def weightedSum(self, A, W):
        ''' Returns a weighted sum of how many agents played any action in collection A (The weight for action A[j] is W[j].) '''
        if len(A) != len(set(A)):
            raise _PosEcInputException(
                "weightSum input A must not contain duplicate actions.")
        if len(A) != len(W):
            raise _PosEcInputException(
                "weightSum inputs A and W must be the same length.")
        return self._access(_PosEcAccessor("SUM", tuple(self.agents), tuple(self.types), None, tuple(A), tuple(W), None))

    def argmax(self, A, fn=lambda x: x, default=None):
        ''' Returns some action a where
(1) a is in A,
(2) a is played by at least one agent, and 
(3) a maximizes fn() subject to (1) and (2).
<default> is returned if no agent plays an action in A.
'''

        A = [(fn(a), a) for a in A]
        A.sort()
        A = [a for v, a in A]
        weights = tuple(range(1, len(A) + 1))
        index = self._access(
            _PosEcAccessor("MAX", tuple(self.agents), tuple(self.types), None, tuple(A), weights, None))
        if index == 0:
            return default
        return A[index - 1]

    def max(self, A, fn=lambda x: x, default=None):
        ''' Returns fn(a) or default (See argmax) '''
        a = self.argmax(A, fn, None)
        if a is None:  # FIXME: What if None is in the map(fn,A)?
            return default
        return fn(a)
        # FIXME: I should be able to make this more efficient by collapsing
        # equal actions together

    def argmin(self, A, fn=lambda x: x, default=None):
        ''' Returns some action in A or default (See argmax) '''
        return self.argmax(A, lambda x: -fn(x), default)

    def min(self, A, fn=lambda x: x, default=None):
        ''' Returns fn(a) or default (See argmax) '''
        a = self.argmin(A, fn, None)
        if a is None:  # FIXME: What if None is in the map(fn,A)?
            return default
        return fn(a)

    def any(self, A, agents=None):
        ''' Returns True iff any agent plays an action in list A
If the agents parameter is not None, it can be a list of agents to consider '''
        if agents is None:
            agents = self.agents
        return self._any(actions=A, agents=agents)

    def plays(self, j, a_j):
        ''' Returns True iff agent j played action a_j
            Optionally, j can be a list or tuple of agents.
        '''
        if isinstance(j, (list, tuple)):
            return self._any(agents=j, actions=[a_j]) == 1
        assert j in self._agg.N
        return self._sum(agents=[j], actions=[a_j]) == 1

    def action(self, j):
        ''' Returns the action played by agent j '''
        if j == self._a.agent:
            return self._a.action
        if self._a.agent is None:
            raise _PosEcInputException(
                "a_N.action() method is not allowed when making k-symmetric games.")
        actions = set([a.action for a in self._agg.A if a.agent == j])
        for act in actions:
            if self.plays(j, act):
                return act
        raise _PosEcInternalException("Agent j does not play any action!")

    def __getitem__(self, j):
        ''' Returns the action of agent j '''
        return self.action(j)


class TypeProfile(_InstrumentedDataStore):

    ''' Representation of an type profile with efficient ways of accessing the data.

A TypeProfile is passed to a Setting's utility function as the argument theta_N
'''

    def hasType(self, j, theta):
        ''' Returns True iff agent j has type theta '''
        if self._a.agent == j:
            return self._a.type == theta
        if not isinstance(self._agg, pyagg.BAGG):
            # Full information game, we can just check his action nodes to find
            # out his type
            return self._agg.S[j][0].type == theta
        return self._sum(agents=[j], types=[theta])

    def type(self, j):
        ''' Returns the type of agent j '''
        for t in self.types:
            if self.hasType(j, t):
                return t
        raise _PosEcInternalException("Agent has no type!")

    def __getitem__(self, j):
        ''' Returns the type of agent j '''
        return self.type(j)

    def count(self, t):
        ''' Returns a count of how many agents have type t '''
        return self.sum([t])

    def sum(self, T):
        ''' Returns a total of how many agents have types in T '''
        if len(T) != len(set(T)):
            # FIXME: Maybe just a warning
            raise _PosEcInputException(
                "sum input T must not contain duplicate types.")
        return self._sum(types=T)

    def weightedSum(self, T, W):
        ''' Returns a weighted sum of how many agents had any type in collection T (The weight for type T[j] is W[j].) '''
        if len(T) != len(set(T)):
            raise _PosEcInputException(
                "weightSum input T must not contain duplicate types.")
        if len(T) != len(W):
            raise _PosEcInputException(
                "weightSum inputs A and W must be the same length.")
        return self._access(_PosEcAccessor("SUM", tuple(self.agents), tuple(T), tuple(W), tuple(self.actions), None, None))

    def any(self, T):
        ''' Returns Truee iff at least one agent has a type in T '''
        if len(T) != len(set(T)):
            # FIXME: Maybe just a warning
            raise _PosEcInputException(
                "sum input T must not contain duplicate types.")
        return self._any(types=T)

    def argmax(self, T, fn=lambda x: x, default=None):
        ''' Returns some type t where
(1) t is in T,
(2) t is the type of at least one agent, and 
(3) t maximizes fn() subject to (1) and (2).
<default> is returned if no agent has any type in T.
'''
        T = [(fn(t), t) for t in T]
        T.sort()
        T = [t for v, t in T]
        weights = tuple(range(1, len(T) + 1))
        index = self._access(
            _PosEcAccessor("MAX", tuple(self.agents), tuple(T), weights, tuple(self.actions), None, None))
        if index == 0:
            return default
        return T[index - 1]

    def max(self, T, fn=lambda x: x, default=None):
        ''' Returns fn(t) or default (See argmax) '''
        t = self.argmax(T, fn, None)
        if t == None:  # FIXME: What if None is in the map(fn,T)?
            return default
        return fn(t)
        # FIXME: I should be able to make this more efficient by collapsing
        # equal actions together

    def argmin(self, T, fn=lambda x: x, default=None):
        ''' Returns some type in T or default (See argmax) '''
        return self.argmax(T, lambda x: -fn(x), default)

    def min(self, T, fn=lambda x: x, default=None):
        ''' Returns fn(t) or default (See argmax) '''
        t = self.argmin(T, fn, None)
        if t == None:  # FIXME: What if None is in the map(fn,T)?
            return default
        return fn(t)


class RealSpace:

    ''' A set-like object that contains vectors of floating-point numbers

Useful for defining the (projected) outcome space in a Setting '''

    def __init__(self, dimensions=None):
        ''' If dimensions==None, it contains all floating point numbers.
If dimensions>=0, it contains all dimensions-length lists & tuples of floating point numbers. '''
        if dimensions is not None:
            assert isinstance(dimensions, int)
            assert dimensions >= 0
        self.dim = dimensions

    def __contains__(self, element):
        ''' If dimensions==None, it contains all floating point numbers.
If dimensions>=1, it contains all dimensions-length lists & tuples of floating point numbers. '''
        if self.dim is None:
            return isinstance(element, (float, int))
        else:
            if not isinstance(element, (list, tuple)):
                return False
            if len(element) != self.dim:
                return False
            return all([isinstance(e, (float, int)) for e in element])

    def __eq__(self, other):
        ''' Returns True iff other is a RealSpace instance with the same dimensions '''
        if not isinstance(other, RealSpace):
            return False
        return self.dim == other.dim

    def __repr__(self):
        if self.dim is None:
            return "RealSpace()"
        return "RealSpace(%d)" % (self.dim)


class CartesianProduct:

    ''' A set-like object that contains vectors where each element in the vector is a member of a "factor" set

Useful for defining the (projected) outcome space in a Setting
'''

    def __init__(self, *factors, **options):
        '''factors is a list of set-like objects (let k denote its length)
contains only k-length vectors where each element is in the corresponding factor

options:
memberType=<type> - contains only objects of type <type> (also supports tuples of types)
'''
        self.factors = factors
        self.memberType = None
        if "memberType" in options.keys():
            self.memberType = options["memberType"]

    def __contains__(self, vector):
        ''' Returns True iff vector is a k-length vector where each element is in the corresponding factor
if memberType!=None, then vector must alos have type <memberType> (
'''
        if len(vector) != len(self.factors):
            return False
        if self.memberType is not None and not isinstance(vector, self.memberType):
            return False
        return all([vector[i] in self.factors[i] for i in range(len(vector))])

    def __eq__(self, other):
        ''' True iff both inputs are CertesianProducts with the same factors and memberType '''
        if not isinstance(other, CartesianProduct):
            return False
        return self.factors == other.factors

    def __repr__(self):
        return "CartesianProduct(*" + repr(self.factors) + ",memberType=" + repr(self.memberType) + ")"


class _TypeCheckDescriptor(object):

    def __init__(self, name, comment, testfn=lambda x: True):
        self.data = {}
        self.name = name
        self.comment = comment
        self.testfn = testfn

    def __get__(self, instance, owner):
        return self.data.get(instance)

    def __set__(self, instance, value):
        if not self.testfn(value):
            raise _PosEcInputException(self.name + " must be " + self.comment)
        self.data[instance] = value


class Setting(object):

    ''' A data structure representing a (not projected) full-information setting '''
    n = _TypeCheckDescriptor(
        "n", "positive integer number of agents", lambda x: isinstance(x, int) and x > 0)
    O = _TypeCheckDescriptor(
        "O", "set-like container of outcomes", lambda x: _hasAttribute(x, "__contains__"))
    Theta = _TypeCheckDescriptor(
        "Theta", "n-length vector of (hash()-able) types", lambda x: isinstance(x, (list, tuple)))
    u = _TypeCheckDescriptor(
        "u", "Utility function; u(i,theta,o,a_i) returns a floating point number\ni is an agent number (0 to n-1), theta is a TypeProfile, o is an outcome and a_i is an action", lambda x: _nArgs(x) >= 4)

    def __init__(self, n, O, Theta, u):
        self.n = n
        self.O = O
        self.Theta = Theta
        self.u = u
        if len(Theta) != n:
            raise _PosEcInputException("Theta must have a length of n")


class ProjectedSetting(Setting):
    Psi = _TypeCheckDescriptor(
        "Psi", "set-like container of projected outcomes", lambda x: _hasAttribute(x, "__contains__"))
    pi = _TypeCheckDescriptor(
        "pi", "Projection function; pi(i,o) returns a projected outcome", lambda x: _nArgs(x) >= 2)
    u = _TypeCheckDescriptor(
        "u", "Projected utility function; u(i,theta,po,a_i) returns a floating point number\ni is an agent number (0 to n-1), theta is a TypeProfile, po is an projected outcome and a_i is an action", lambda x: _nArgs(x) >= 4)

    # FIXME: I wish I had something a bit more like this, but it won't work with symmetry etc...
    # Basically, I want to benefit from whatever projection wherever I can, if it's only half the agents, I still want it.
    # Psi = _TypeCheckDescriptor(
    #    "Psi", "n-length vector of set-like containers of projected outcomes", lambda x: isinstance(x,(list,tuple)) and all([_hasAttribute(s,"__contains__") for s in x]))

    def __init__(self, n, O, Theta, u, Psi, pi):
        self.n = n
        self.O = O
        self.Theta = Theta
        self.u = u
        self.Psi = Psi
        self.pi = pi


class BayesianSetting(Setting):
    P = _TypeCheckDescriptor(
        "P", "n-length vector of Distributions over Theta", lambda x: isinstance(x, (list, tuple)))
    Theta = _TypeCheckDescriptor(
        "Theta", "Finite collection of (hash()-able) types", lambda x: isinstance(x, (list, tuple)))

    def __init__(self, n, O, Theta, P, u):
        self.n = n
        self.O = O
        self.Theta = Theta
        self.P = P
        self.u = u
        if len(P) != n:
            raise _PosEcInputException("P must have a length of n")


class ProjectedBayesianSetting(ProjectedSetting, BayesianSetting):

    def __init__(self, n, O, Theta, P, u, Psi, pi):
        self.n = n
        self.O = O
        self.Theta = Theta
        self.P = P
        self.u = u
        self.Psi = Psi
        self.pi = pi
        if len(P) != n:
            raise _PosEcInputException("P must have a length of n")


class Mechanism(object):

    A = _TypeCheckDescriptor(
        "A", "Action function; A(setting, i, theta_i) returns a collection of (hash()-able) actions")
    M = _TypeCheckDescriptor(
        "M", "Outcome function; M(setting, a_N) returns an outcome or a Distribution over outcomes")

    def __init__(self, A, M):
        self.A = A
        self.M = M


class ProjectedMechanism(Mechanism):

    ''' FIXME: Consider not passing the theta_i; it's kind of inappropriate '''
    M = _TypeCheckDescriptor(
        "M", "Outcome function; M(setting, i, theta_i, a_N) returns a (projected) outcome or a Distribution over (projected) outcomes")
    
_PosEcActionNode = namedtuple("PosEcActionNode", ["agent", "type", "action"])


class Distribution(object):

    ''' Representation of a distribution with finite support

Used to represent type probabilities in a BayesianSetting and randomized outcomes in a Mechanism '''

    def __init__(self, support, probabilities):
        ''' support and probabilities are equal-length vectors '''
        self.support = support
        self.probabilities = probabilities

    def __iter__(self):
        ''' Returns an iterator over 2 tuples representing the probability of an event and the event '''
        for i in range(len(self.support)):
            yield self.support[i], self.probabilities[i]
        # FIXME: Maybe that's backwards

    def probability(self, event):
        ''' Returns the probability of event
Returns 0.0 if event is not in support '''
        if event not in self.support:
            return 0.0
        return self.probabilities[self.support.index(event)]


class UniformDistribution(Distribution):

    def __init__(self, support):
        self.support = support
        n = len(support)
        self.probabilities = [1.0 / n] * n


def _findDependencies(setting, mechanism, agg, quiet=False, addAnonymity=False):
    dependencies = []

    def addDependency(agg, a, f):
        if (a, f) in dependencies:
            raise _PosEcInternalException(
                "Duplicate Arc Adding! " + repr((a, f)))

        if f not in agg.F and isinstance(f, _PosEcAccessor):
            if f.function == "SUM":
                agg.F.append(f)
                agg.f[f] = (pyagg.FN_TYPE_WEIGHTED_SUM, 0)
                for a2 in agg.A:
                    add = True
                    if a2.agent is not None and a2.agent not in f.agents:
                        add = False
                    if a2.type not in f.types:
                        add = False
                    if a2.action not in f.actions:
                        add = False
                    if add:
                        weight = 1
                        if f.action_weights is not None:
                            weight = f.action_weights[
                                list(f.actions).index(a2.action)]
                        elif f.type_weights is not None:
                            weight = f.type_weights[
                                list(f.types).index(a2.type)]
                        agg.v.append((a2, f, weight))
            elif f.function == "MAX":
                agg.F.append(f)
                agg.f[f] = (pyagg.FN_TYPE_WEIGHTED_MAX, 0)
                for a2 in agg.A:
                    add = True
                    if a2.agent is not None and a2.agent not in f.agents:
                        add = False
                    if a2.type not in f.types:
                        add = False
                    if a2.action not in f.actions:
                        add = False
                    if add:
                        weight = 1
                        if f.action_weights is not None:
                            weight = f.action_weights[
                                list(f.actions).index(a2.action)]
                        elif f.type_weights is not None:
                            weight = f.type_weights[
                                list(f.types).index(a2.type)]
                        agg.v.append((a2, f, weight))
            else:
                raise _PosEcInternalException(
                    "Can't handle a required function!")

        agg.v.append((f, a))
        dependencies.append((a, f))
        agg._makeArcDict()
        agg._makeEffectDict()

    if not quiet:
        logging.info("Finding missing arcs:")
    count = 0
    if addAnonymity:
        if not quiet:
            logging.info("Adding anonymity arcs.")
        actions = tuple(set([a.action for a in agg.A]))
        agents = tuple(set([a.agent for a in agg.A]))
        types = tuple(set([a.type for a in agg.A]))
        accessors = []
        for a in actions:
            accessor = _PosEcAccessor(
                function="SUM", agents=agents, types=types, type_weights=None, actions=(a,), action_weights=(1,), signature=None)
            accessors.append(accessor)
        for act in agg.A:
            for accessor in accessors:
                addDependency(agg, act, accessor)
    for a in agg.A:
        if not quiet:
            logging.info(a)
        while True:
            finished = True
            agg._makeArcDict()
            agg._makeEffectDict()
            C = agg._possibleConfigs(a)
            if not C:
                C.append(tuple())
            for c in C:
                try:
                    utility = agg.u[a](c)
                except KeyError:
                    key = sys.exc_info()[1][0]
                    # print sys.exc_info()
                    # print "Missing Arc",a,key
                    count += 1
                    # if not quiet:
                        # print count
                    addDependency(agg, a, key)
                    finished = False
                    break
            if finished:
                break
        if not quiet:
            logging.info()
    if not quiet:
        logging.info("Done.")


class _PayoffFn(object):

    def __init__(self, setting, mechanism, a, transform=None):
        self.setting = setting
        self.mechanism = mechanism
        self.a = a
        self.transform = transform

    def u(self, o, theta_N):
        a = self.a
        if isinstance(self.setting, ProjectedSetting):
            if o not in self.setting.O and o not in self.setting.Psi:
                raise _PosEcInputException("Invalid outcome: " + repr(o))
            if o in self.setting.O:
                o = self.setting.pi(self.a.agent, o)
        else:
            if o not in self.setting.O:
                raise _PosEcInputException("Invalid outcome: " + repr(o))
        if self.transform is not None:
            return self.transform(self.setting, a.agent, theta_N, o, a.action)
        else:
            return self.setting.u(a.agent, theta_N, o, a.action)

    def __call__(self, config):
        a = self.a
        #inArcs = [f for f in self.agg.F if self.agg._hasArc(f, a)]
        inArcs = [arc for arc in self.agg.v if arc[1] == a]
        if len(inArcs) != len(config):
            neighbours = [(n)
                          for n in self.agg.A + self.agg.F if self.agg._hasArc(n, a)]
            print inArcs, config, len(inArcs), len(config), neighbours
            raise _PosEcInternalException(
                "Length mismatch between depedencies and configurations")
        # print "***",inArcs,config
        # for arc in self.agg.v:
        #    print "Arc",arc
        neighbours = self.agg.neighbours(self.a)
        # for n,c in zip(neighbours, config):
        #    print n,c
        a_N = ActionProfile(self.agg, a, dict(zip(neighbours, config)))
        a_N.actions = tuple(set([anode.action for anode in self.agg.A]))
        a_N.agents = range(self.setting.n)
        a_N.types = tuple(set(self.setting.Theta))
        theta_N = TypeProfile(self.agg, a, dict(zip(neighbours, config)))
        theta_N.agents = range(self.setting.n)
        theta_N.types = tuple(set(self.setting.Theta))
        theta_N.actions = tuple(set([anode.action for anode in self.agg.A]))
        a_N.config = inArcs, config
        if isinstance(self.mechanism, ProjectedMechanism):
            lottery = self.mechanism.M(self.setting, a.agent, a.type, a_N)
        else:
            lottery = self.mechanism.M(self.setting, a_N)
        if not isinstance(lottery, Distribution):
            # A bit of syntactic sugar
            lottery = UniformDistribution([lottery])
        return sum([(self.u(o, theta_N) * p) for o, p in lottery])


def _hashable(o):
    try:
        hash(o)
        return True
    except:
        return False


def structureInference(setting, mechanism, agg, quiet=False, bbsi_level=0, metrics=None):
    if metrics is None:
        metrics = dict()
    if not quiet:
        start = time.time()
    _findDependencies(setting, mechanism, agg, quiet, bbsi_level == 2)
    metrics['WBSI-time'] = time.time() - start
    metrics['BNFG-size'] = agg.sizeAsNFG()
    metrics['Post-WBSI size'] = agg.sizeAsAGG()
    if not quiet:
        logging.info("WBSI time: %s" % metrics['WBSI-time'])
        logging.info("BNFG size: %s" % metrics['BNFG-size'])
        logging.info("Post-WBSI size:" % metrics['Post-WBSI size'])
    if bbsi_level > 0:
        bbsi.preprocess(agg)
        if bbsi_level == 2:
            if not quiet:
                logging.info("Performing anonymity-favoring cuts.")
                start = time.time()
            bbsi.anonymityCuts(agg)
            metrics['anon-size'] = agg.sizeAsAGG()
            metrics['anon-time'] = time.time() - start
            if not quiet:
                logging.info("Post-anonymity-cut size:", metrics['anon-size'])
                logging.info("anonymity-cut time:", metrics['anon-time'])

        if not quiet:
            logging.info("General BBSI:")
        bbsi.compressByILS(agg)
        if not quiet:
            logging.info("Done.")
        metrics['BBSI-size'] = agg.sizeAsAGG()
        metrics['BBSI-time'] = time.time() - start
        if not quiet:
            logging.info("Post-BBSI size:", metrics['BBSI-size'])
            logging.info("BBSI time:", metrics['BBSI-time'])
    return agg


def makeAGG(setting, mechanism, symmetry=False, transform=None, quiet=False, bbsi_level=0, metrics=None):
    ''' Takes a Setting and a Mechanism, returns a corresponding pyagg.AGG object

transform is a function (setting,i,theta_i,a_i,o,theta_N) that returns a real value
quiet==True produces no standard output
bbsi_level==0 means to do no BBSI
bbsi_level==1 means to do limited BBSI (suitable for fine-tuning games)
bbsi_level==2 means to do extensive BBSI (suitable for discovering coarse structure)
'''
    # FIXME: bbsi==1 is still way too time-consuming for large games - I need
    # to refine these parameters.
    if metrics is None:
        metrics = dict()

    assert isinstance(setting, Setting)
    if isinstance(setting, BayesianSetting):
        return _makeBAGG(setting, mechanism, symmetry, transform, quiet, bbsi_level, metrics)

    assert bbsi_level in [0, 1, 2]
    _N = range(setting.n)
    _A = []
    _S = {}
    if symmetry:
        for theta in set(setting.Theta):
            aTheta = [_PosEcActionNode(None, theta, a)
                      for a in mechanism.A(setting, None, theta)]
            _A += aTheta
            for i in _N:
                if setting.Theta[i] == theta:
                    _S[i] = aTheta
    else:
        for i in _N:
            theta_i = setting.Theta[i]
            _S[i] = [_PosEcActionNode(i, theta_i, a)
                     for a in mechanism.A(setting, i, theta_i)]
            _A += _S[i]

    for a in _A:
        if not _hashable(a.action):
            raise _PosEcInputException(
                "Actions must be hash()-able: " + repr(a.action))
    for theta in setting.Theta:
        if not _hashable(theta):
            raise _PosEcInputException(
                "Types must be hash()-able: " + repr(theta))

    _F = []
    _v = []
    _f = {}
    _u = {}

    for a in _A:
        _u[a] = _PayoffFn(setting, mechanism, a, transform)
    agg = pyagg.AGG(_N, _A, _S, _F, _v, _f, _u)
    for a in _A:
        _u[a].agg = agg
    return structureInference(setting, mechanism, agg, quiet=quiet, bbsi_level=bbsi_level, metrics=metrics)

_PosEc_BAGG_type = namedtuple("_PosEc_BAGG_type", ("agent", "type"))


def _makeBAGG(setting, mechanism, symmetry=False, transform=None, quiet=False, bbsi_level=1, metrics=None):
    ''' Takes a BayesianSetting and a Mechanism, returns a corresponding pyagg.BAGG

transform is a function (setting,i,theta_i,a_i,o,theta_N) that returns a real value
quiet==True produces no standard output
bbsi_level==0 means to do no BBSI
bbsi_level==1 means to do limited BBSI (suitable for fine-tuning games)
bbsi_level==2 means to do extensive BBSI (suitable for discovering coarse structure)
'''
    assert isinstance(setting, BayesianSetting)
    if metrics is None:
        metrics = dict()

    _N = range(setting.n)
    _A = []
    _S = {}

    _P = {}
    for i in _N:
        _P[i] = {}
    if symmetry:
        _Theta = [_PosEc_BAGG_type(None, theta) for theta in setting.Theta]
        for theta in _Theta:
            aTheta = [_PosEcActionNode(None, theta.type, a)
                      for a in mechanism.A(setting, None, theta)]
            _A += aTheta
            _S[theta] = aTheta
            for i in _N:
                _P[i][theta] = setting.P[i].probability(theta.type)
    else:
        _Theta = [_PosEc_BAGG_type(i, theta)
                  for theta in setting.Theta for i in _N if setting.P[i].probability(theta) > 0.0]
        for theta in _Theta:
            aTheta = [_PosEcActionNode(theta.agent, theta.type, a)
                      for a in mechanism.A(setting, None, theta)]
            _A += aTheta
            _S[theta] = aTheta
            for i in _N:
                if i == theta.agent:
                    _P[i][theta] = setting.P[i].probability(theta.type)
                else:
                    _P[i][theta] = 0.0
    for a in _A:
        if not _hashable(a.action):
            raise _PosEcInputException(
                "Actions must be hash()-able: " + repr(a.action))
    for theta in setting.Theta:
        if not _hashable(theta):
            raise _PosEcInputException(
                "Types must be hash()-able: " + repr(theta))
    _F = []
    _v = []
    _f = {}
    _u = {}

    for a in _A:
        _u[a] = _PayoffFn(setting, mechanism, a, transform)
    agg = pyagg.BAGG(_N, _Theta, _P, _A, _S, _F, _v, _f, _u)
    for a in _A:
        _u[a].agg = agg
    return structureInference(setting, mechanism, agg, quiet=quiet, bbsi_level=bbsi_level, metrics=metrics)

'''
    Things I'm currently missing:
-Lots of good input checking
-implement all my existing position auction stuff
-implement all my voting generators - DONE but not fully tested 
-implement RDS for voting - DONE but not fully tested
'''


def explain(agg, acts=None):
    ''' Describes the set of function-calls that can be used to produce a
strategically equivalent AGG.
Optionally, acts can cover a specific subset of action nodes.
'''
    allTypes = set([a.type for a in agg.A])
    allActions = set([a.action for a in agg.A])
    allAgents = set(agg.N)

    def _explainFn(agg, fn):
        ''' Outputs a command that will compute what is necessary for the computation '''
        if set(fn.actions) == allActions:
            prefix = "theta."
            if fn.function == "SUM" and set(fn.agents) == allAgents:
                if set(fn.type_weights) == set([1]):
                    if len(fn.types) == 1:
                        return prefix + "count(%r)" % fn.types
                    else:
                        return prefix + "sum(%r)" % (fn.types,)
                else:
                    return prefix + "weightedSum(%r,%r)" % (fn.types, fn.type_weights)
            if fn.function == "MAX" and set(fn.agents) == allAgents:
                if fn.type_weights != None:
                    return prefix + "max(%r,fn=??)" % (fn.types,)
                return prefix + "any(%r)" % (fn.types,)
            if len(fn.types) == 1 and len(fn.agents) == 1:
                return prefix + "plays(%r,%r)" % (fn.agents[0], fn.types[0])
        else:
            prefix = "a_N."
            if fn.function == "SUM" and set(fn.agents) == allAgents:
                if set(fn.action_weights) == set([1]):
                    if len(fn.actions) == 1:
                        return prefix + "count(%r)" % fn.actions
                    else:
                        return prefix + "sum(%r)" % (fn.actions,)
                else:
                    return prefix + "weightedSum(%r,%r)" % (fn.actions, fn.action_weights)
            if fn.function == "MAX" and set(fn.agents) == allAgents:
                if set(fn.action_weights) != set([1]):
                    return prefix + "max(%r,fn=??)" % (fn.actions,)
                return prefix + "any(%r)" % (fn.actions,)
            if len(fn.actions) == 1 and len(fn.agents) == 1:
                return prefix + "plays(%r,%r)" % (fn.agents[0], fn.actions[0])
            if len(fn.actions) == 1:
                return prefix + "plays(%r,%r)" % (fn.agents, fn.actions[0])
        return fn
    if acts is None:
        acts = agg.A
    for a in acts:
        print a
        for fn in [v[0] for v in agg.v if v[1] == a]:
            print "  ", _explainFn(agg, fn)
        print
