from itertools import permutations
from math import factorial, dist, sqrt

ans = []

while True: 
    n = int(input())
    if n == 0:
        break
    w = list(map(int, input().split()))
    
    dp = [[0 for i in range(n+1)] for j in range(n+1)]
    
    for i in range(n-1): 
        if abs(w[i] - w[i+1]) < 2:
            dp[i][2] = 2
    
    for j in range(3,n+1): 
        for i in range(n-2): 
            if i+j >= n: 
                continue
            if abs(w[i] - w[i+j]) < 2 : 
                dp[i][j] = dp[i+1][j-2] + 2
            if abs(w[i] - w[i+1]) < 2 : 
                dp[i][j] = max(dp[i][j], dp[i+2][j-2] + 2)
            if abs(w[i+j-1] - w[i+j]) < 2 : 
                dp[i][j] = max(dp[i][j], dp[i][j-2] + 2)
            for k in range(2,j): 
                dp[i][j] = max(dp[i][j], dp[i][k] + dp[i+k][j-k])
                
    ans.append(dp[0][n])
    
for _ in ans: 
    print(_)
    
    
    
    





