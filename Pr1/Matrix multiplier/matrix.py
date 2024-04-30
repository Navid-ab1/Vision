def mul(A, B):
    row_A, col_A = len_check(A)
    row_B, col_B = len_check(B)
    C = [[0 for i in range(col_B)] for j in range(row_A)]
    if col_A == row_B:
        for i in range(row_A):
            for j in range(col_B):
                for k in range(row_B):
                    C[i][j] += (A[i][k] * B[k][j])

        return C

    else:
        C.clear()
        return C


def len_check(A):
    row = len(A)
    num = A[0]
    column = len(num)
    return row, column


A = [[1, 0, 0], [0, 0, 3], [0, 2, 0]]
B = [[1, 1], [0, .5], [2, 1 / 3.0]]
C = [[1, 0, 0], [0, 0, 0.5], [0, 1 / 3.0, 0]]

print(mul(A, B))
print(mul(B, A))
print(mul(A, C))
