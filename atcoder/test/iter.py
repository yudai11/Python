
import time
start = time.time()

for i in range(1000000):
        i ** 2

elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time))
