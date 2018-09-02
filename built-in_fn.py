# coding:utf-8
"""
Test for built-in function
"""

import os


class A(object):
    bar = 1


class model(object):
    def __init__(self):
        self.a = 11

    def call(self,s):
        print self.a , s


def main2():
    # 这个只有在新式类中才有的，对于对象的所有特性的访问，都将会调用这个方法来处理。。。可以理解为在__getattr__之前
    class Fjs(object):
        def __init__(self, name):
            self.name = name

        def hello(self):
            print "said by : ", self.name

        def __getattribute__(self, item):
            print "访问了特性：" + item
            return object.__getattribute__(self, item)

    fjs = Fjs("fjs")
    print fjs.name
    fjs.hello()

def main1():
    a = A()
    print getattr(a, 'bar')
    # AttributeError: 'A' object has no attribute 'bar2'
    # getattr(a, 'bar2')

    print getattr(a, 'bar2', 3)  # bar2 not exist, but set a default value
    print getattr(a, 'bar', 2)

    m = model()
    m('s')


def main3():
    class Fjs(object):
        def __init__(self, name):
            self.name = name

        def hello(self):
            print "said by : ", self.name

        def fjs(self, name):
            if name == self.name:
                print "yes"
            else:
                print "no"

    class Wrap_Fjs(object):
        def __init__(self, fjs):
            self._fjs = fjs

        # 在访问对象的item属性的时候，如果对象并没有这个相应的属性，方法，那么将会调用这个方法来处理。。。
        # 这里要注意的时，假如一个对象叫fjs,  他有一个属性：fjs.name = "fjs"，
        # 那么在访问fjs.name的时候因为当前对象有这个属性，那么将不会调用__getattr__()方法，而是直接返回了拥有的name属性了
        def __getattr__(self, item):
            if item == "hello":
                print "调用hello方法了"
            elif item == "fjs":
                print "调用fjs方法了"
            return getattr(self._fjs, item)

    fjs = Wrap_Fjs(Fjs("fjs"))
    fjs.hello()
    fjs.fjs("fjs")


if __name__ == '__main__':
    main3()