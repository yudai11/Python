from itertools import permutations
from math import factorial, dist, sqrt
from collections import Counter
from bisect import bisect_right, bisect_left, insort_left, insort_right
import math



n = int(input())
_s = input()
s = []
for i in range(n):
    s.append(int(_s[i]))

ans = 0

for i in range(n):
    ans += ((pow(10, n-i) - 1) / 9) * s[i]

print("{:d}".format(ans))
