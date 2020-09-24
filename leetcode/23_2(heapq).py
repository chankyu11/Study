import heapq

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        head = curr = ListNode()
        heap = []
        
        for i in range(len(lists)):
            if lists[i]:
                heapq.heappush(heap, (lists[i].val, i, lists[i]))
                
        while heap:
            node = heapq.heappop(heap)
            index = node[1]
            curr.next = node[2]
            
            curr = curr.next
            if curr.next:
                heapq.heappush(heap, (curr.next.val, index, curr.next))
                
        return head.next
                               
        
    
    

                
        
        
        