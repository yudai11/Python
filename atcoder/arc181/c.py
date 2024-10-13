def generate_grid(N, P, Q):
    # Initialize an empty N x N grid
    grid = [[0 for i in range(N)] for j in range(N)]
    
    # To satisfy the condition SP1<SP2<⋯<SPN, we'll generate rows in increasing order
    for i in range(N):
        for j in range(N):
            if i < j:
                grid[P[i] - 1][j] = 0
            else:
                grid[P[i] - 1][j] = 1

    # To satisfy the condition TQ1<TQ2<⋯<TQN, we'll adjust columns accordingly
    for j in range(N):
        col = [grid[i][Q[j] - 1] for i in range(N)]
        col.sort()
        for i in range(N):
            grid[i][Q[j] - 1] = col[i]
    
    return grid

# Example usage
N = 3
P = [3, 1, 2]
Q = [2, 3, 1]
grid = generate_grid(N, P, Q)
print(grid)
