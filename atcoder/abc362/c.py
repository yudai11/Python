from itertools import permutations
from math import factorial, dist, sqrt

n = int(input())

l = []
r = []

lower = 0
upper = 0

for i in range(n):
    li, ri = map(int, input().split())
    l.append(li)
    r.append(ri)
    lower += li
    upper += ri
    
if lower > 0 or upper < 0:
    print("No")
else :
    print("Yes")
    
    x = [l[i] for i in range(n)]

    for i in range(n): 
        x[i] = min(r[i], -lower + l[i])
        lower += x[i] - l[i]
        if lower >= 0:
            break
        
    for i in range(n):
        print(x[i], end = " ")
