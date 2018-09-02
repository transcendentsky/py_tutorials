# -*- coding: utf-8 -*-
"""
Simple demo for python lib collections
"""

import collections

Point = collections.namedtuple('Point', ['x', 'y'])
p = Point(1,2)

print p

print isinstance(p, Point)

