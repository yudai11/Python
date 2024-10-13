from itertools import permutations
from math import factorial, dist, sqrt

n = int(input())
s = input()

s_int = [0 for i in range(n)]
for i in range(n):
    if s[i] == 'S':
        s_int[i] = 1
    if s[i] == 'P':
        s_int[i] = 2

dp = [[0 for i in range(2)] for j in range(n)]

dp[0][1] = 1
    
for i in range(1,n):
    for j in range(2):
        if j == 0:
            if s_int[i - 1] == s_int[i]:
                dp[i][j] = dp[i - 1][1]
            elif (s_int[i - 1] - s_int[i] + 2) % 3 == 0:
                 dp[i][j] = dp[i - 1][0]
            else :
                dp[i][j] = max(dp[i - 1][0], dp[i - 1][1])
        else :
            if s_int[i - 1] == s_int[i]:
                dp[i][j] = dp[i - 1][0] + 1
            elif (s_int[i - 1] - s_int[i] + 1) % 3 == 0:
                 dp[i][j] = dp[i - 1][1] + 1
            else :
                dp[i][j] = max(dp[i - 1][0], dp[i - 1][1]) + 1
      
ans = max(dp[n-1][0], dp[n-1][1])          
print(ans)
        
