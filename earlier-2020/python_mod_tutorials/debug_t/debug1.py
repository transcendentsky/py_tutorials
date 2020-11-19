#coding:utf-8
"""
c: (continue)继续执行
w:(words)显示当前行的上下文信息
a:(arguments)打印当前函数的参数列表
s:(stop)执行当前行，并在顶一个可能的时机停止
n:(next)继续执行直到当前函数的下一行或者函数返回值

break 或 b  : 设置断点
list / l   : 查看当前行的代码段
exit / q   : 中止并退出
pp         : 打印变量的值
help       : 帮助
"""

import pdb

def make_bread():
    ###
    pdb.set_trace()
    return "I don't have time"

print(make_bread())


"""
使用 PyCharm 进行调试

PyCharm 是由 JetBrains 打造的一款 Python IDE，具有语法高亮、Project 管理、代码跳转、智能提示、自动完成、单元测试、版本控制等功能，
同时提供了对 Django 开发以及 Google App Engine 的支持。分为个人独立版和商业版，需要 license 支持，也可以获取免费的 30 天试用。
试用版本的 Pycharm 可以在官网上下载，下载地址为：http://www.jetbrains.com/pycharm/download/index.html。 
PyCharm 同时提供了较为完善的调试功能，支持多线程，远程调试等，可以支持断点设置，单步模式，表达式求值，变量查看等一系列功能。
PyCharm IDE 的调试窗口布局如图 1 所示
"""

"""
使用日志功能达到调试的目的

日志信息是软件开发过程中进行调试的一种非常有用的方式，特别是在大型软件开发过程需要很多相关人员进行协作的情况下。
开发人员通过在代码中加入一些特定的能够记录软件运行过程中的各种事件信息能够有利于甄别代码中存在的问题。
这些信息可能包括时间，描述信息以及错误或者异常发生时候的特定上下文信息。 
最原始的 debug 方法是通过在代码中嵌入 print 语句，通过输出一些相关的信息来定位程序的问题。
但这种方法有一定的缺陷，正常的程序输出和 debug 信息混合在一起，给分析带来一定困难，当程序调试结束不再需要 debug 输出的时候，
通常没有很简单的方法将 print 的信息屏蔽掉或者定位到文件。python 中自带的 logging 模块可以比较方便的解决这些问题，
它提供日志功能，将 logger 的 level 分为五个级别，可以通过 Logger.setLevel(lvl) 来设置。默认的级别为 warning。
"""