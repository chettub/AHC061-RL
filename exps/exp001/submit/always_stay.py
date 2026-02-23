from __future__ import annotations

import sys


def main() -> None:
    data = sys.stdin
    line = data.readline()
    if not line:
        return
    n, m, t_max, u_max = map(int, line.split())
    v = [list(map(int, data.readline().split())) for _ in range(n)]
    sx = [0] * m
    sy = [0] * m
    for p in range(m):
        sx[p], sy[p] = map(int, data.readline().split())

    # current position of player0
    x0, y0 = sx[0], sy[0]

    for _ in range(t_max):
        print(x0, y0, flush=True)

        # tx_0..tx_{m-1}
        for _ in range(m):
            data.readline()

        # ex_0..ex_{m-1}
        for p in range(m):
            x, y = map(int, data.readline().split())
            if p == 0:
                x0, y0 = x, y

        # owner grid
        for _ in range(n):
            data.readline()
        # level grid
        for _ in range(n):
            data.readline()


if __name__ == "__main__":
    main()

