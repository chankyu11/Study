class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        g = collections.defaultdict(list)
        
        for a, b in sorted(tickets):
            g[a].append(b)
        
        tmp = []
        def dfs(a):
            while g[a]:
                dfs(g[a].pop(0))
            tmp.append(a)
        
        dfs('JFK')
        return tmp[::-1]
                
            
            
