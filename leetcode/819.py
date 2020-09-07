
class Solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        
        words = re.sub('[^\w]',' ', paragraph).lower().split()
        # print(words)
        words = filter(lambda x: x not in banned, words)
        print(words)
                                         
        counts = collections.Counter(words)
        print(counts)
        return counts.most_common(1)[0][0]                                 
        