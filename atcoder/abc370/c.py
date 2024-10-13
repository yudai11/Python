s = list(input())
t = list(input())

# if s > t:
#     tmp = s
#     s = t
#     t = tmp
    
cp = s
x = []
cnt = 0
for i in range(len(s)):
    if s[i] > t[i]:
        cp[i] = t[i]
        x.append("".join(cp))
        cnt += 1
        
for i in range(len(s)-1,-1,-1):
    if cp[i] != t[i]:
        cp[i] = t[i]
        x.append("".join(cp))
        cnt += 1

print(cnt)
for i in range(len(x)):
    print(x[i])
