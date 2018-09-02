# -*- coding:utf-8 -*-
import yaml
import numpy as np
import sys
import os
import subprocess
try:
    import thread
    print(sys.version)
except:
    import _thread as thread
import time


def printwocao(o=None):
    i = 0
    while i < 11:
        print('wocao? {}'.format(o))
        time.sleep(1)
        i += 1

thread.start_new_thread(printwocao,())

# 这个十分重要， 当主进程结束时，子线程也会结束
while 1:
    pass