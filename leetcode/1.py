class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:       
        nd = {}
        
        for i in range(len(nums)):
            if target - nums[i] not in nd:
                nd[nums[i]] = i
            else:
                return [nd[target - nums[i]], i]
            
            
            
#         for i in range(len(nums)):
#             for j in range(len(nums) + 1):
#                 if nums[i] + nums[j] == target:
#                     return [i, j]
                