

class Solution:
    def isPalindrome(self, s: str) -> bool:
        strs = []
        for c in s:
            if c.isalnum():
                strs.append(c.lower())
            
        
        while len(strs) > 1:
            if strs.pop(0) != strs.pop():
                return False
            
        return True
        