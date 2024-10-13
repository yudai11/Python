import sys

n, t = map(int, input().split())
a = list(map(int, input().split()))

tate = [0 for _ in range(n)]
yoko = [0 for _ in range(n)]
naname = [0] * 2

for k in range(t): 
    i = (a[k] - 1) // n
    j = (a[k] - 1) % n
    tate[i] += 1
    
    if tate[i] >= n:
        print(k+1)
        sys.exit()
        
    yoko[j] += 1
    if yoko[j] >= n:
        print(k+1)
        sys.exit()
        
    if i == j :
        naname[0] += 1
    if i + j == n - 1 :
        naname[1] += 1
    if naname[0] >= n or naname[1] >= n:
        print(k+1)
        sys.exit()
        
print(-1)
    
