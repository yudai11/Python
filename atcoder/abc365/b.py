from itertools import permutations
from math import factorial, dist, sqrt

n = int(input())
a = list(map(int, input().split()))

max_loc = 0
second_max_loc = 0
if a[0] < a[1]:
    max_loc = 1
else :
    second_max_loc = 1
    
for i in range(2,n):
    if a[i] > a[max_loc]:
        second_max_loc = max_loc
        max_loc = i
    elif a[max_loc] > a[i] and a[i] > a[second_max_loc]:
        second_max_loc = i

print(second_max_loc + 1)


