from itertools import permutations
from math import factorial, dist, sqrt
from collections import Counter
from bisect import bisect_right, bisect_left, insort_left, insort_right

t = int(input())
ans = [False for i in range(t)]

for i in range(t):
    s = input()
    x = input()
    y = input()
    
    s_len = len(s)
    s_array = [s[i] for i in range(s_len)]
    int_x = [0 for _ in range(len(x))]
    int_y = [0 for _ in range(len(y))]
    
    num_0_in_x, num_0_in_y, num_1_in_x, num_1_in_y = 0, 0, 0, 0
    
    for i in range(len(x)):
        if x[i] == '0':
            num_0_in_x += 1
        else:
            num_1_in_x += 1
            int_x[i] = 1
            
    for i in range(len(y)):
        if y[i] == '0':
            num_0_in_y += 1
        else:
            num_1_in_y += 1
            int_y[i] = 1
            
    if num_1_in_y - num_1_in_x == 0:
        fx = s * num_0_in_x
        fy = s * num_0_in_y
        if fx == fy:
            ans[i] = True
        break
            
    t_len = (num_0_in_x - num_0_in_y) * s_len // (num_1_in_y - num_1_in_x)
            
    if (num_0_in_x - num_0_in_y) * s_len % (num_1_in_y - num_1_in_x) != 0:
        break
    elif t_len < 0:
        break
    
    is_decided_t = [False for _ in range(t_len)]
    t_array = [i for i in range(t_len)]
    
    fx_array, fy_array = [], []
    
    for i in range(len(int_x)):
        if int_x[i] == 0:
            fx_array = fx_array + s_array
        else:
            fx_array = fx_array + t_array
            
    for i in range(len(int_y)):
        if int_y[i] == 0:
            fy_array = fy_array + s_array
        else:
            fy_array = fy_array + t_array
            
    for i in range(len(fx_array)):
        if str.isdigit(fx_array[i]) and not is_decided_t[fx_array[i]] and not str.isdigit(fy_array[i]):
            j = fx_array[i]
            t[i] = fy_array[i]
            is_decided_t[fx_array[i]] = True
        if str.isdigit(fy_array[i]) and not is_decided_t[fy_array[i]] and not str.isdigit(fx_array[i]):
            j = fy_array[i]
            t[i] = fx_array[i]
            is_decided_t[fy_array[i]] = True
    
    t = ""
    for i in range(t_len):
        t = t + str(t_array[i])
        
    fx, fy = "", ""
    
    for i in range(len(x)):
        if int_x[i] == 0:
            fx = fx + s
        else:
            fx = fx + t
            
    for i in range(len(y)):
        if int_y[i] == 0:
            fy = fy + s
        else:
            fy = fy + t
            
    if fx == fy:
        ans[i] = True
        
for i in range(t):
    if ans[i]:
        print("Yes")
    else:
        print("No")
            
            
    
    
    
    