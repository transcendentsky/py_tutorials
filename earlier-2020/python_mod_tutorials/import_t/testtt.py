#coding:utf-8
import sys

# run this file
# to test import function
print("Python version", sys.version_info)

# Code below failed to run
# if sys.version_info[0] == 2:
#     try:
#         import module_a
#         from module_a import a_num
#     except:
#         raise EnvironmentError("Python version?? ")
# else:
#     try: # For python 3
#         import .module_a
#         from .module_a import a_num
#     except:
#         print(sys.version_info)

# Right code
try:
    from . import module_a   # Python 3 , Python 2 不行
    import module_a
    from .module_a import a_num
except ImportError:
    import module_a
    from module_a import a_num

print(a_num)