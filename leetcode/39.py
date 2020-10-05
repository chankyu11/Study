class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        tmp = []
        
        def dfs(idx, total):
            if total > target:
                return
            if total == target:
                res.append(tmp[::])
                return
            
            for i in range(idx, len(candidates)):
                total += candidates[i]
                tmp.append(candidates[i])
                
                dfs(i, total)
                tmp.pop()
                total -= candidates[i]
        total = 0
        dfs(0, total)
        return res
        