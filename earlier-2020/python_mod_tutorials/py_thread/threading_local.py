# -*- coding: utf-8 -*-
"""
threading.local 是为了方便在线程中使用全局变量而产生
每个线程的全局变量互不干扰

"""

from threading import local, enumerate, Thread, currentThread

local_data = local()
local_data.name = 'local_data'


class TestThread(Thread):
    def run(self):
        print currentThread()
        print local_data.__dict__
        local_data.name = self.getName()
        local_data.add_by_sub_thread = self.getName()
        print local_data.__dict__


if __name__ == '__main__':
    print currentThread()
    print local_data.__dict__
    print '----------------'

    t1 = TestThread()
    t1.start()
    t1.join()
    print '----------------'

    t2 = TestThread()
    t2.start()
    t2.join()
    print '----------------'

    print currentThread()
    print local_data.__dict__