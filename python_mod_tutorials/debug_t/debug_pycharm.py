#!/usr/bin/python
#coding:utf-8
import thread
import time


# Define a function for the thread
def print_time(threadName, delay):
  count = 0
  while count < 5:
    count += 1
    print "%s: %s" % (threadName, time.ctime(time.time()))


def check_sum(threadName, valueA, valueB):
  print "to calculate the sum of two number her"
  result = sum(valueA, valueB)
  print "the result is", result


def sum(valueA, valueB):
  if valueA > 0 and valueB > 0:
    return valueA + valueB


def readFile(threadName, filename):
  file = open(filename)
  for line in file.readlines():
    print line


try:
  thread.start_new_thread(print_time, ("Thread-1", 2,))
  thread.start_new_thread(check_sum, ("Thread-2", 4, 5,))
  thread.start_new_thread(readFile, ("Thread-3", "test.txt",))
except:
  print "Error: unable to start thread"
while 1:
  # 	 print "end"
  pass
