# -*- coding:utf-8 -*-

"""
 进程间通信（IPC，Inter-Process Communication）

>>> 本地socket链接
import socket
import os

if __name__ == '__main__':
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if os.path.exists("/tmp/test.sock"):
        os.unlink("/tmp/test.sock")
    server.bind("/tmp/test.sock")
    server.listen(0)
    while True:
        connection, address = server.accept()
        connection.send("test: %s"% connection.recv(1024))
    connection.close()


import socket
import os

if __name__ == '__main__':
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect("/tmp/test.sock")

    instr = raw_input()
    client.send(instr)
    print client.recv(1024)

    client.close()
"""


import socket
import os

if __name__ == '__main__':
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("localhost", 8888))
    server.listen(0)
    while True:
        connection, address = server.accept()
        connection.send("test: %s"% connection.recv(1024))
    connection.close()


import socket
import os

if __name__ == '__main__':
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("localhost", 8888))

    instr = raw_input()
    client.send(instr)
    print(client.recv(1024))

    client.close()