class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        look = {"(": ")", "{": "}", "[": "]"}
        for i in s:
            if i in look:
                stack.append(i)
            elif len(stack) == 0 or look[stack.pop()] != i:
                return False
            
        return len(stack) == 0
            
    