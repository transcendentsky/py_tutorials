#coding:utf-8
"""
Test for built-in function
"""

import os

class A(object):
  bar = 1


if __name__ == "__main__":
  a = A()
  print getattr(a, 'bar')
  # AttributeError: 'A' object has no attribute 'bar2'
  # getattr(a, 'bar2')

  print getattr(a, 'bar2', 3) # bar2 not exist, but set a default value
  print getattr(a, 'bar', 2)
