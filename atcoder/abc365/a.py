from itertools import permutations
from math import factorial, dist, sqrt

y = int(input())

if y % 4 != 0 :
    print(365)
elif y % 100 != 0 :
    print(366)
elif y % 400 == 0:
    print(366)
else :
    print(365)
    