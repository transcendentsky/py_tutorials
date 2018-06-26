#coding:utf-8

class Student(object):
    def __init__(self):
        self._birth = 1 # 如果初始化了self._birth 则可以直接输出，否则会报错没有这个值

    @property
    def birth(self):
        return self._birth

    @birth.setter
    def birth(self, value):
        self._birth = value

    @property
    def age(self):
        return 2015 - self._birth

st = Student()
# 需要先设置 st.birth 不然会报错
# st.birth = 10
print(st.birth)
