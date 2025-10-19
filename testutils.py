def rounded(matrix):
    if matrix.dim() == 0:
        raise ValueError("Cannot round empty")
    if matrix.dim() == 1:
        return [round(x, 2) for x in matrix.tolist()]
    r = []
    for v in matrix:
        r.append(rounded(v))
    return r
