class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if len(digits) == 0:
            return []
        
        dic = {0: "0",
               1: "1",
               2: "abc", 
               3: "def", 
               4: "ghi", 
               5: "jkl",
               6: "mno", 
               7: "pqrs", 
               8: "tuv", 
               9: "wxyz"}
        res = [""]
        
        for dig in digits:
            tmp = []
            for cha in dic[int(dig)]:
                for str in res:
                    tmp.append(str + cha)
                    
            res = tmp
        return res