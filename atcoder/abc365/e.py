from itertools import permutations
from math import factorial, dist, sqrt

n = int(input())

l = [0 for _ in range(n)]
r = [0 for _ in range(n)]

for i in range(n):
    l[i], r[i] = map(int, input().split())
    
q = int(input())

coords = [[0 for i in range(4)] for j in range(q)]

for i in range(q):
    for j in range(4):
        coords[i] = map(int, input().split())
        
ans = [0 for i in range(q)]



