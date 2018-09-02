#coding:utf-8
from queue import Queue
from threading import Thread
import time

# A thread that produces data
def producer(out_q):
    while True:
        # Produce some data
        # ...
        data = 'a str'
        out_q.put(data)

# A thread that consumes data
def consumer(in_q):
    while True:
        # Get some data
        data = in_q.get()
        print(data)
        # Process the data


# Create the shared queue and launch both threads
q = Queue()
# t1 = Thread(target=consumer, args=(q,))
# t2 = Thread(target=producer, args=(q,))
# t1.start()
# t2.start()


def printer():
    for _ in range(12):
        time.sleep(1)
        print("testing...")
    print('over...')

t3 = Thread(target=printer, args=())
t3.start()    # 线程开始
time.sleep(2)
print("start join")
t3.join()     # 等待程序完成
print('Join Over..')