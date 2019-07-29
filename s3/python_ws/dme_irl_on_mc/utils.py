"""
  @author: yigit.yildirim@boun.edu.tr
"""

from collections import namedtuple

Step = namedtuple('Step', 'state action')


class State(object):
    def __init__(self, x=0, v=0):
        self.x = x
        self.v = v

    # euler
    def distance(self, s):
        return ((s.x - self.x)**2 + (s.v - self.v)**2)**0.5


class Action(object):
    def __init__(self, f=0):
        self.force = f


class ObjectworldState(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    # euler
    def distance(self, s):
        return ((s.x - self.x)**2 + (s.y - self.y)**2)**0.5


class ObjectworldAction(object):
    def __init__(self, xch, ych):  # xch, ych imply intended changes in the state. e.g. DOWN implies -1 change in y dir
        self.xch = xch
        self.ych = ych
