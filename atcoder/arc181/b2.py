def can_find_T(S, X, Y):
    if len(X) != len(Y):
        return False

    # Initialize T as an empty string
    T = ""

    for x, y in zip(X, Y):
        if x == y:
            continue
        elif x == '0' and y == '1':
            T += S
        elif x == '1' and y == '0':
            T += S

    # Generate f(S, T, X) and f(S, T, Y) and compare
    def generate_f(S, T, pattern):
        result = ""
        for char in pattern:
            if char == '0':
                result += S
            else:
                result += T
        return result

    f_X = generate_f(S, T, X)
    f_Y = generate_f(S, T, Y)

    return f_X == f_Y

# Example usage
S = "araara"
X = "01"
Y = "111"
print(can_find_T(S, X, Y))  # 出力: True または False
