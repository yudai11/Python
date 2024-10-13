import math

a, b, x = map(int, input().split())

if x >= a ** 2 * b / 2:
    tan_theta = 2 * (a * a * b - x) / (a * a * a)
    ans = math.degrees(math.atan(tan_theta))
    print(ans)
else:
    tan_theta = b ** 2 * a / 2 / x
    ans = math.degrees(math.atan(tan_theta))
    print(ans)