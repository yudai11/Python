# -*- using:utf-8 -*-
import time

def gcd_python(x, y):
    if y == 0:
        return x
    else:
        return gcd_python(y, x % y)
    
dump = []
max_iter = 100000
for i in range(max_iter):
    start = time.time()
    
    gcd_python(10000,325)
    
    elapsed_time = time.time() - start
    dump.append(elapsed_time)

print(sum(dump)/max_iter)