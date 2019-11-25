"""
Test functions included built-in functions,

"""

class Cls(object):
  def __init__(self, *args, **kwargs):
    self.var1 = 1
    self.var2 = 2
    print("Execute __init__")

  def __call__(self, func):
    print("Execute __call__")
    return func


# Cls('1')
# Cls(1)

@Cls()
def pp():
  print("print pp")

pp()
Cls()(1)

if isinstance([1], (str, tuple, list)):
  print 'OK'


an = [None] * 10
print an