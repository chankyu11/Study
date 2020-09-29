import collections
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        freqs = collections.Counter(nums)
        heap = []
        
        for f in freqs:
            # print(f)
            heapq.heappush(heap,(-freqs[f], f))
        
        res = []
        for _ in range(k):
            res.append(heapq.heappop(heap)[1])
            # print(res)
        return res