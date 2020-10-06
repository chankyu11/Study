class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        
        def dfs(nums, temp, i):
            res.append(temp[:])
            
            for i in range(i, len(nums)):
                temp.append(nums[i])
                dfs(nums, temp, i + 1)
                
                temp.pop()
        dfs(nums, [], 0)
        return res