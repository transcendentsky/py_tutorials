# -*- coding:utf-8 -*-
'''''示例3: 使用语法糖@来装饰函数，相当于“myfunc = deco(myfunc)”
但发现新函数只在第一次被调用，且原函数多调用了一次'''


def deco(func):
  print("before myfunc() called.")
  func()
  print("  after myfunc() called.")
  return func


@deco
def myfunc():
  print(" myfunc() called.")

print("[###] 这时还未执行函数")

# 也就是说装饰器是一段执行代码， 在导入包时执行，将其包装变成另一个函数

myfunc()


print("--------- 分割线 ------------")
'''''示例4: 使用内嵌包装函数来确保每次新函数都被调用， 
内嵌包装函数的形参和返回值与原函数相同，装饰函数返回内嵌包装函数对象'''

# 这种在内部的就不一样了， 这种就是在调用时启动， 原来是这样
def deco2(func):
  def _deco():
    print("before myfunc() called.")
    func()
    print("  after myfunc() called.")
    # 不需要返回func，实际上应返回原函数的返回值
  return _deco

@deco2
def myfunc2():
  print(" my func2() called ")


myfunc2()