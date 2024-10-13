n = int(input())
a = list(map(int, input().split()))
for i in range(n):
    a[i] = a[i] -1

b = [0] * n

for i in range(n):
    b[a[i]] = i
  
num_change = 0
ans_1 = []
ans_2 = []

for i in range(n):
    if b[i] != i:
        num_change = num_change + 1
        tmp = b[i]
        b[i] = i
        b[a[i]] = tmp
        ans_1.append(min(b[i], b[a[i]]) + 1)
        ans_2.append(max(b[i], b[a[i]]) + 1)
        a[tmp] = a[i]
        a[i] = i
    
print(num_change)
    
if num_change != 0:
    for i in range(num_change):
        print(ans_1[i], end = " ")
        print(ans_2[i])