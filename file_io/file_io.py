# coding:utf-8
"""
r	以只读方式打开文件。文件的指针将会放在文件的开头。这是默认模式。

rb	以二进制格式打开一个文件用于只读。文件指针将会放在文件的开头。
    这是默认模式。一般用于非文本文件如图片等。
r+	打开一个文件用于读写。文件指针将会放在文件的开头。

rb+	以二进制格式打开一个文件用于读写。文件指针将会放在文件的开头。一般用于非文本文件如图片等。

w	打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。
    如果该文件不存在，创建新文件。
wb	以二进制格式打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。
    如果该文件不存在，创建新文件。一般用于非文本文件如图片等。
w+	打开一个文件用于读写。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。
    如果该文件不存在，创建新文件。
wb+	以二进制格式打开一个文件用于读写。如果该文件已存在则打开文件，并从开头开始编辑，
    即原有内容会被删除。如果该文件不存在，创建新文件。一般用于非文本文件如图片等。
a	打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。
    也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。
ab	以二进制格式打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。
    也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。
a+	打开一个文件用于读写。如果该文件已存在，文件指针将会放在文件的结尾。
    文件打开时会是追加模式。如果该文件不存在，创建新文件用于读写。
ab+	以二进制格式打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。
     如果该文件不存在，创建新文件用于读写。
"""
from __future__ import print_function
import os
import sys

"""
def find_previous(run):
    output_dir = cfg.EXP_DIR

    if not os.path.exists(os.path.join(output_dir, 'run' + str(run) + '_checkpoint_list.txt')):
        return False
    with open(os.path.join(output_dir, 'run' + str(run) + '_checkpoint_list.txt'), 'r') as f:
        lineList = f.readlines()
    epoches, resume_checkpoints = [list() for _ in range(2)]
    for line in lineList:
        # print("line:", line.find('epoch '), line.find(':'))
        epoch = int(line[line.find('epoch ') + len('epoch '):line.find(':')])
        checkpoint = line[line.find(':') + 2:-1]
        epoches.append(epoch)
        resume_checkpoints.append(checkpoint)
    return epoches, resume_checkpoints
"""
if not os.path.exists('checkpoint_list.txt'):
  pass

with open('checkpoint_list.txt', 'r') as f:
  abspath = os.path.abspath('.')
  print(abspath)
  linelist = f.readlines()
epoches, resume_checkpoints = list(), list()
for line in linelist:
  epoch = int(line[line.find('epoch ') + len('epoch '): line.find(':')])
  checkpoint = line[line.find(':') + 2:-1]
  epoches.append(epoch)
  resume_checkpoints.append(checkpoint)
# return epoches, resume_checkpoints


with open('checkpoint_list.txt', 'r') as f:
  abspath = os.path.abspath('.')
  print(abspath)
  lines = f.readlines()
  for line in lines:
    line = line.strip('\n')  # 去除末尾的 回车
    mark = line.find('!')
    print(mark < 0)
    print(line, end=' 2333 \n', file=sys.stderr)

################################################
##  介绍一下标准输出
# print(*objects, sep=' ', end='\n', file=sys.stdout)

##  当文件为空时，返回空集
with open('Nothing.py', 'r') as f:
  lines = f.readlines()
  print(lines == None)
  print(lines)

# split 的返回测试
sss = 'python tt.py'
fff = 'ipconfig'
s_s = sss.split(' ')
s_f = fff.split(' ')
print(s_s)
print(s_f)
# print(type(s_s))
assert type(s_s) is list, '?'
assert type(s_f) is list, '??'
