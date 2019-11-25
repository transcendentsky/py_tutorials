#coding:utf-8

# 作为库，下面的导入为 .tt 而不是 tt,
# 但是这样作为主程序启动就会报错，
# "ImportError: attempted relative import with no known parent package"
#
# 如果用.tt 作为主程序不会报错，但作为库使用会报错
# ImportError: No module named 'tt'
# 如果作为库，可以添加 -m
# python -m file_io.print_abspath （要从主文件夹启动）

# 不管从哪里输出，当前文件夹总是主程序文件夹

import os
from .tt import print_dir

def test():
    print_dir.test()

if __name__ == '__main__':
    print_dir.test()
    s = os.path.abspath('.')
    print("file_io.print abspath",s)

