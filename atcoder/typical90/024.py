from itertools import permutations
from math import factorial, dist, sqrt
from collections import Counter
from bisect import bisect_right, bisect_left, insort_left, insort_right

n, k = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

test = 0
for i in range(n):
    test = test + abs(a[i] - b[i])

if test <= k and (k - test) % 2 == 0:
    print("Yes")
else:
    print("No")