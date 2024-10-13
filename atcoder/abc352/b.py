s = input()
t = input()


j = 0
len_t = len(t)

for i in range(len(s)):
    while j < len_t:
        if t[j] == s[i]:
            j += 1
            print(j, end = " ")
            break
        j += 1
        
print()
            

