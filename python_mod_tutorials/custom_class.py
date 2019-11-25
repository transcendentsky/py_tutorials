#coding:utf-8
class Student(object):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return 'Student object (name: %s)' % self.name

# 类似的还有 __repr__, __iter__ , __next__, 等等，应该视object函数的 “复写”
# __call__ 