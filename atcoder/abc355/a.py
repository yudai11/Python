from itertools import permutations
from math import factorial, dist, sqrt
from collections import Counter
from bisect import bisect_right, bisect_left, insort_left, insort_right

a, b = map(int, input().split())

if a != b:
    print(6 - a - b)
else:
    print(-1)