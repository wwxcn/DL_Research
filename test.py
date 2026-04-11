'''
给定一个m x n 网格，每个格子上都有金币（0<=c<100），
机器人从左到右走，可以往右、右上、右下移动，问从最左侧都到最右侧，能获得的最大金币数是多少
[[2, 4, 7, 2, 1],
[5, 0, 4, 3, 3],
[1, 6, 2, 1, 6],
[4, 4, 5, 8, 0]]

ans[i, j] = maze[i, j] + max(ans[i - 1, j], ans[i-1, j-1], ans[i - 1, j+1])


'''

class solution():
    def maxCorn(maze):
        n, m = len(maze), len(maze[0])
        ans = [[0 for _ in range(n)], [0 for _ in range(n)]]
        cur = 0
        ans[cur, :] = maze[:, 0]
        for i in range(m):
            cur = 1 - cur
            for j in range(n):
                ans[cur][j] = max(ans[cur][j], maze[j, i] + ans[1 - cur][j - 1]) if j > 0 else ans[cur][j]
                ans[cur][j] = max(ans[cur][j], maze[j, i] + ans[1 - cur][j + 1]) if j < n - 1 else ans[cur][j]
                ans[cur][j] = max(ans[cur][j], maze[j, i] + ans[1 - cur][j])
        
        return max(ans[cur])

