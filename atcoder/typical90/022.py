a  = list(map(int, input().split()))

def gcd_2 (x, y):
    a = [x, y]
    while True:
        a.sort()
        m = a[1] % a[0]
        if m == 0:
            return a[0]
        if m == 1:
            return 1
        a[1] = m
        
gcd = gcd_2(a[0], gcd_2(a[1], a[2]))
ans = 0
for i in range(3):
    ans += a[i] // gcd - 1
print(ans, end="\n")