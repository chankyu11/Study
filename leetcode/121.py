import sys

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        pro = 0
        min_p = sys.maxsize
        
        for price in prices:
            min_p = min(min_p, price)
            pro = max(pro, price - min_p)
        return pro
            
    


        
