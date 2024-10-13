from itertools import permutations
from math import factorial, dist, sqrt
from collections import Counter
from bisect import bisect_right, bisect_left, insort_left, insort_right

mod = 998244353

n = int(input())
copy_n = n

digit_n = 1

while True:
    copy_n //= 10
    if copy_n == 0:
        break
    digit_n += 1

copy_n = n

# V_n = (10^{(n+1)d} - 1) / (10^d - 1) mod
copy_n %= mod


V_denu = pow(10, (n * digit_n) % (mod-1), mod) - 1

x = pow(10, digit_n, mod) - 1
x %= mod

V_deno = pow(x, mod - 2, mod)

V_n = (n % mod) * (V_denu * V_deno) % mod

print(V_n)
