from itertools import permutations
from math import factorial, dist, sqrt
from collections import Counter
from bisect import bisect_right, bisect_left, insort_left, insort_right




def count_point(a, b : int, diam : int, n: int):
    min_val = b - diam
    max_val = b + diam
    min_loc = 0
    max_loc = n-1
    upper = n-1
    lower = 0
    while True:
        test = lower + (upper - lower) // 2
        if test == lower:
            min_loc = upper
            break
        if a[test] >= min_val:
            upper = test
        else:
            lower = test
    upper = n-1
    lower = min_loc
    while True:
        test = lower + (upper - lower) // 2
        if test == lower:
            max_loc = lower
            break
        if a[test] <= max_val:
            lower = test
        else:
            upper = test
            
    return (max_loc - min_loc +1)

n, q = map(int, input().split())
a = list(map(int, input().split()))
a.sort()

query = [[0 for i in range(2)] for j in range(q)]

for i in range(q):
    query[i] = map(int, input().split())
    
ans = [0 for _ in range(q)]
    
max = 2 * pow(10, 8)


        

for i in range(q):
    b, k = query[i]
    lower = -1
    upper = max
    while True:
        test = lower + (upper - lower) // 2
        if test == lower:
            ans[i] = lower + 1
            break
        if bisect_right(a, b + test) - bisect_left(a, b - test) + 1 <= k:
            lower = test
        else:
            upper = test
            
for i in range(q):
    print(ans[i])
    
    