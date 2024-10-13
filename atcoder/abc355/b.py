from itertools import permutations
from math import factorial, dist, sqrt
from collections import Counter
from bisect import bisect_right, bisect_left, insort_left, insort_right

n, m = map(int, input().split())

a = list(map(int, input().split()))
a.sort()
b = list(map(int, input().split()))
b.sort()
b.append(300)

feasi = False

j = 0
for i in range(n-1):
    while True:
        if j == m:
            break
        if b[j] < a[i] and j < m:
            j += 1
        else:
            break
    
    if b[j] > a[i+1]:
        feasi = True
        break
    
    
if feasi and n > 1:
    print("Yes")
else:
    print("No")
