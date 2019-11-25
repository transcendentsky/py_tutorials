#coding:utf-8
print("BBBBBBBBBBBBB")

print("B import C")

try:
    import module_c   # Python 2 在这里是可以的， 但是Python 3 不行， 我佛，
except ImportError:
    import submodule.module_c  # Python 3 要这样？？？
    from . import module_c
# import submodule.module_c   # Python 2,3 在这里都是可以的