#coding:utf-8
import threading
import time
import subprocess
import sys
if sys.version_info[0] > 2:
    import queue
else:
    import Queue as queue

class TaskRunning(threading.Thread):
    def __init__(self, task_q, max_process=1):
        self.num_process = 0
        self.max_process = max_process
        self.process_list = []

        self.task_q = task_q
        self.process_q = queue.LifoQueue(maxsize=max_process)
        self.running = 1
        super(TaskRunning, self).__init__()

    def run(self):
        self.running = 1
        while self.running:
            time.sleep(1)
            self.check_process()
            if self.task_q.qsize() > 0:
                # if self.process_q.qsize() < self.max_process:
                    # task_command = self.task_q.get()
                    # obj = subprocess.Popen([task_command], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    # obj = subprocess.Popen([task_command])
                    # self.process_q.put(obj)
                print("ooooooo")
            print("[!] ooooooo")

    def check_process(self):
        length = self.process_q.qsize()
        for i in range(length):
            obj = self.process_q.get()
            if obj.poll() == None:
                self.num_process -= 1
            else:
                self.process_q.put(obj)

    def join(self):
        self.running = 0
        super(TaskRunning, self).join()

q = queue.LifoQueue(maxsize=10)
tr = TaskRunning(q)
tr.start()
time.sleep(3)
tr.join()
print("tr joined, and restart")
# 并不能再启动
tr.start()   # RuntimeError: threads can only be started once
print("?")


# 测试subprocess的程序启动
# import os
# kaishi
# a = os.path.dirname(__file__)
# s = os.path.join(a,'py_thread.py')
# print(a)
# obj = subprocess.Popen(["python",s])
# time.sleep(2)
# obj.join()
# print('????')
# out, err = obj.communicate()
# print(out)