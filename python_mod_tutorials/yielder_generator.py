#coding:utf-8

def gensquares(N):
    try:
        yield 99
        # for i in range(N):
        #     yield i**2
    finally:
        print("Finally yield")

for item in gensquares(3):
    print item,

# 首先，生成器的好处是延迟计算，一次返回一个结果。
# 也就是说，它不会一次生成所有的结果，这对于大数据量处理，将会非常有用。

# 因此，生成器的唯一注意事项就是：生成器只能遍历一次。
