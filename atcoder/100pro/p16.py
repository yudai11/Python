from itertools import permutations
from math import factorial, dist, sqrt

n = int(input())
p = tuple(map(int, input().split()))
q = tuple(map(int, input().split()))

a = b = 0
i = 0

for perm in permutations(range(1, n+1)):
    if perm == p: 
        a = i 
    if perm == q:
        b = i
    i += 1
    
print(abs(a -b))

