#coding:utf-8
"""
装饰器　与　类装饰器

# 编写类装饰器
类装饰器类似于函数装饰器的概念，但它应用于类，它们可以用于管理类自身，或者用来拦截实例创建调用以管理实例。

"""


### 单体类
# 由于类装饰器可以拦截实例创建调用，所以它们可以用来管理一个类的所有实例，或者扩展这些实例的接口。
# 下面的类装饰器实现了传统的单体编码模式，即最多只有一个类的一个实例存在。

instances = {}  # 全局变量，管理实例

def getInstance(aClass, *args):
  if aClass not in instances:
    instances[aClass] = aClass(*args)
  return instances[aClass]  # 每一个类只能存在一个实例


def singleton(aClass):
  print("Execute before Class")
  def onCall(*args):
    return getInstance(aClass, *args)

  return onCall


# 为了使用它，装饰用来强化单体模型的类：
# 可以把类所有代码看作一个整体
# 然后返回一个创建类的函数， 相当与把类变成一个函数

@singleton  # Person = singleton(Person)
class Person:
  def __init__(self, name, hours, rate):
    self.name = name
    self.hours = hours
    self.rate = rate

  def pay(self):
    return self.hours * self.rate

@singleton  # Spam = singleton(Spam)
class Spam:
  def __init__(self, val):
    self.attr = val


print(instances)

bob = Person('Bob', 40, 10)
print(bob.name, bob.pay())

sue = Person('Sue', 50, 20)
print(sue.name, sue.pay())

X = Spam(42)
Y = Spam(99)
print(X.attr, Y.attr)

### 第二种， 拦截类里的所有方法
# 其实可以简单的看作为
# @Tracer(class Spam) 这种
# 然后返回仍然是一个类， 然后通过 __getattr__ 拦截
# 还可以实现更多复杂的功能

def Tracer(aClass):
  class Wrapper:
    def __init__(self, *args, **kargs):
      self.fetches = 0
      self.wrapped = aClass(*args, **kargs)

    def __getattr__(self, attrname):
      print('Trace:' + attrname)
      self.fetches += 1
      return getattr(self.wrapped, attrname)

  return Wrapper


@Tracer
class Spam:
  def display(self):
    print('Spam!' * 8)


@Tracer
class Person:
  def __init__(self, name, hours, rate):
    self.name = name
    self.hours = hours
    self.rate = rate

  def pay(self):
    return self.hours * self.rate
