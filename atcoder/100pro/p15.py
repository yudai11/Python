from itertools import permutations
from math import factorial, dist, sqrt

n = int(input())

coords = []

for _ in range(n):
    x, y = map(float, input().split())
    coords.append((x,y))
    
sum_val = sum(
    dist(perm[i], perm[i+1]) for i in range(n-1) for perm in permutations(coords)
)

mean_val = sum_val / factorial(n)
print(mean_val)

