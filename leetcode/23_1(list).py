import heapq

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        
        curr = None
        head = curr
        arr = []
        for lst in lists:
            while lst:
                arr.append(lst.val)
                lst = lst.next
                

        for val in sorted(arr):
            if not curr:
                head = curr = ListNode(val)
            else:
                curr.next = ListNode(val)
                curr = curr.next
        return head
    

                
        
        
        