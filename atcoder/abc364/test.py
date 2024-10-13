from itertools import permutations
from math import factorial, dist, sqrt
from collections import Counter
from bisect import bisect_right, bisect_left, insort_left, insort_right


a = [3, 6, 100, 1, 2]
a.sort()
print(bisect_right(a, 1))
print(bisect_left(a, 1))