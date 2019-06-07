"""
  @author: yigit.yildirim@boun.edu.tr
"""

from collections import namedtuple

Step = namedtuple('Step', 'state action')


class State(object):
    def __init__(self, x=0, v=0):
        self.x = x
        self.v = v


class Action(object):
    def __init__(self, v=0):
        self.v = v
