
try:
    a = 7/0
    print float(a)
except BaseException as e:
    print e.message