"""
  @author: yigit.yildirim@boun.edu.tr
"""


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


class Object(object):
    def __init__(self, location=ObjectworldState(0, 0), inner_color=0, outer_color=0):
        self.location = location
        self.inner_color = inner_color
        self.outer_color = outer_color
