#coding:utf-8
"""
先尝试执行try， 失败则执行except
最后都会执行 finally
"""
try:
  raise NotImplementedError("test")
except:
  print("print except")
finally:
  print("print finally")

try:
  print("print try")
  # raise NotImplementedError("test")
except:
  print("print except")
finally:
  print("print finally")

try:
    fh = open("testfile", "w")
    fh.write("这是一个测试文件，用于测试异常!!")
except IOError:
    print "Error: 没有找到文件或读取文件失败"
else:
    print "内容写入文件成功"
    fh.close()