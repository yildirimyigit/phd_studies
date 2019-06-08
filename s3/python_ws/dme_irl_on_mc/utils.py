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
