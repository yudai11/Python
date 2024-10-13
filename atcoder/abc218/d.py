from itertools import permutations
from math import factorial, dist, sqrt
from collections import Counter
from bisect import bisect_right, bisect_left, insort_left, insort_right



n = int(input())
xy = []

for i in range(n):
    x,y = map(int, input().split())
    xy.append((x,y))
    
xy_set = set(xy)
    
ans = 0
for i in range(n-1):
    for j in range(i+1, n):
        if xy[i][0] == xy[j][0] or xy[i][1] == xy[j][1]:
            continue
        if ((xy[i][0], xy[j][1]) in xy_set) and ((xy[j][0], xy[i][1]) in xy_set):
            ans += 1

print(ans // 2)