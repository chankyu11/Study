# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        h = head
        
        while h and h.next:
            h.val , h.next.val = h.next.val, h.val
            h = h.next.next
        return head