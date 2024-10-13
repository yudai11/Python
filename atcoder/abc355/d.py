n = int(input())
l = [0] * n
r = [0] * n

for i in range(n):
    l[i], r[i] = map(int, input().split())
    
l.sort()
r.sort()

ans = n * (n - 1) // 2
j = 0
for i in range(n):
    while r[j] < l[i]:
        j += 1
    ans -= j
print(ans)
