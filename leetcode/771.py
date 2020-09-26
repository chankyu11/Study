class Solution:
    def numJewelsInStones(self, J: str, S: str) -> int:
        jew = 0
        for s in S:
            if s in J:
                jew += 1
        return jew