# -*- coding: utf-8 -*-
"""
测试imp模块
tensorflow中使用 imp 模块导入特定库（应该是c++库 ）

"""


import imp
import sys

# 使用 imp 导入c++ 库
# 参数1 为名称， 参数2 为路径， 在指定路径寻找
fn_, path, desc = imp.find_module('mymodule', ['./'])
print fn_,path,desc
mod = imp.load_module("mymodule", fn_, path, desc)
print dir(mod)

#这样就会把/data/module/mymodule.py模块导入进来，load_modul方法的第一个参数可以任意写，例如mym,

#作用相当于 import mymodule as mym