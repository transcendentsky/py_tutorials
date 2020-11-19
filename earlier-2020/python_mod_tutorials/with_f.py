#!/usr/bin/env python
#coding:utf-8
# with_example01.py

class Sample:
    def __enter__(self):
        print "In __enter__()"
        return "Foo"

    def __exit__(self, type, value, trace):
        print "In __exit__()"

            
def get_sample():
    return Sample()


with get_sample() as sample:
    print "sample:", sample



class Sample2:
    def __enter__(self):
        return self

    def __exit__(self, type, value, trace):
        print "type:", type
        print "value:", value
        print "trace:", trace
        
    def do_something(self):
        bar = 1/0
        return bar + 10

with Sample2() as sample:
    sample.do_something()

"""
这个例子中，with后面的get_sample()变成了Sample()。
这没有任何关系，只要紧跟with后面的语句所返回的对象有 __enter__()和__exit__()方法即可。
此例中，Sample()的__enter__()方法返回新创建的Sample对象，并赋值给变量sample。

实际上，在with后面的代码块抛出任何异常时，__exit__()方法被执行。
正如例子所示，异常抛出时，与之关联的type，value和stack trace传给__exit__()方法，
因此抛出的ZeroDivisionError异常被打印出来了。
开发库时，清理资源，关闭文件等等操作，都可以放在__exit__方法当中。
"""