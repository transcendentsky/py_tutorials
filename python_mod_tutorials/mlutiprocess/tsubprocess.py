import subprocess
import time

# obj = subprocess.Popen(['python' ,'tsleep.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
obj = subprocess.Popen(['python' ,'tsleep.py'])
# out,err = obj.communicate()
# print(out)
# obj.wait(2)

# print(obj.poll())

# time.sleep(2)
while True:
    time.sleep(2)
    if obj.poll() is None:
        #
        print("Running , ", obj.poll())
    else:
        print('Over pid', obj.poll())
        obj.terminate()

