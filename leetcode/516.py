class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        rs = [ s[len(s)- i - 1] for i in range(len(s))]
        p = [[0 for _ in range(len(s)+1)] for _ in range(len(s)+1)]

        for si in range(1, len(s)+1):
            for rsi in range(1, len(s)+1):
                if s[si-1] == rs[rsi-1]:
                    p[si][rsi] = p[si - 1][rsi - 1] + 1
                else:
                    p[si][rsi] = max(p[si][rsi-1], p[si-1][rsi])

        return p[len(s)][len(s)]


sol = Solution()
x = sol.longestPalindromeSubseq("bbbab")
print(x)