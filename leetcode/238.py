class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        out = []
        p = 1
        
        for i in range(len(nums)):
            out.append(p)
            p = p * nums[i]
            # print(out)
        p = 1
        for i in range(len(nums) - 1, -1, -1):
            # print(i)
            out[i] = out[i] * p
            p = p * nums[i]
            # print(out)
        return out
        
        

        