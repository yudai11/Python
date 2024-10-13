from itertools import permutations
from math import factorial, dist, sqrt
from collections import Counter
from bisect import bisect_right, bisect_left, insort_left, insort_right

n, t, a = map(int, input().split())
if t > n // 2 or a > n // 2:
    print("Yes")
else: 
    print("No")