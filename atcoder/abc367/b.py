from itertools import permutations
from math import factorial, dist, sqrt
from collections import Counter
from bisect import bisect_right, bisect_left, insort_left, insort_right

x = list(input())
y = x    
    
exist_period = False
for i in range(len(x)):
    if x[i] == '.':
        exist_period = True
        
if exist_period:
    while x[-1] == '0':
        x.pop()
    if x[-1] == '.':
        x.pop()
    
print("".join(x))