from itertools import permutations
from math import factorial, dist, sqrt
from collections import Counter
from bisect import bisect_right, bisect_left, insort_left, insort_right

n = int(input())

s = []
s_len = []
max_len = 0

for i in range(n):
    si = input()
    max_len = max(max_len, len(si))
    s_len.append(len(si))
    s.append(si)
    
    
t_len = [0 for i in range(max_len)]
for i in range(n):
    x = len(s[n - i - 1])
    for j in range(x):
        t_len[j] = i+1
    
ans = []

for i in range(max_len):
    ti = ""
    for j in range(t_len[i]):
        if i < s_len[n - j - 1] :
            ti += s[n - j - 1][i]
        else:
            ti += "*"
    ans.append(ti)
    
for i in range(max_len):
    print(ans[i])