# Longest palindromic substring
class Solution:
    def longestPalindrome(self, s: str) -> str:
        p = [[1 for i in range(len(s))] for j in range(len(s))]
        lenbest = 1
        left = 0
        right = 0

        for j in range(1, len(s)):
            for front in range(0, len(s) - j):
                back = j + front
                if front == back - 1:
                    if s[front] == s[back]:
                        p[front][back] = 2
                    else: p[front][back] = 0
                elif front < back - 1:
                    if s[front] == s[back]:
                        if p[front + 1][back - 1] > 0:
                            p[front][back] = 2 + p[front+1][back-1]
                        else: p[front][back] = 0
                    else:
                        p[front][back] = 0

                if p[front][back] >= lenbest:
                    left = front
                    right = back
                    lenbest = p[front][back]
                    print(left, right, lenbest)

        print(s[left:right+1])
        print(left, right)
        print(p)
        return s[left:right+1]


class Solution2:
    def longestPalindrome(self, s: str) -> str:
        rs = [ s[len(s) - 1 - i] for i in range(len(s))]
        p = [ [0 for i in range(len(s))] for j in range(len(s))]
        lenbest = 0
        left = 0
        right = 0

        for rsi in range(len(s)):
            for si in range(len(s)):
                if rs[rsi] != s[si]:
                    p[rsi][si] = 0
                else:
                    if rsi == 0 or si == 0:
                        p[rsi][si] = 1
                    else:
                        p[rsi][si] = p[rsi-1][si-1] + 1

                if p[rsi][si] >= lenbest:
                    right = si
                    left = si - p[rsi][si] + 1
                    lenbest = p[rsi][si]

            print(s[left:right + 1])
            print(left, right, p[rsi][si], lenbest)
            print(p)
        return s[left:right+1]






sol = Solution2()
store = ""
sol.longestPalindrome("ccd")