class Solution:
    def calc(self, n: int, m: int) -> int:
        if n > 100000 or n < 1 or m > 1000000000 or m < 1:
            raise ValueError
        c = 0
        while m >= n:
            m -= n
            c += 1
            # print("re", c, n)
        if m == 0:
            # print("????", c, n)
            return c
        elif m < n:
            # print("?", c, n)
            return c + self.calc(n-1, m)
        else:
            raise ValueError("")


sol = Solution()
x = sol.calc(4,10)
print(x)