from itertools import permutations
from math import factorial, dist, sqrt

n, m = map(int, input().split())
a = list(map(int, input().split()))



lower = 0
upper = max(a)

if sum(a) <= m :
    print("infinite")
else:
    while True:
        test = (upper - lower) // 2 + lower
        if test == lower:
            break
        sum_val = 0
        for i in range(n):
            sum_val += min(test, a[i])
        if sum_val > m:
            upper = test
        else:
            lower = test
    print(lower)
    
