from itertools import permutations
from math import factorial, dist, sqrt

n, a, b = map(int, input().split())
d = list(map(int, input().split()))

mod = a + b
day_array = set({})

for i in range(n):
    tmp = d[i] % mod
    day_array.add(tmp)
    
day_array = list(day_array)

day_array.sort()
k = len(day_array)

diff = abs(day_array[0] - day_array[k-1] + mod)
for i in range(k-1):
    diff = max(diff, day_array[i+1] - day_array[i])

if diff > b:
    print("Yes")
else:
    print("No")