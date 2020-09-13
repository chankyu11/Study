# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode(0)
        res = dummy
        carry = 0
        while l1 or l2:
            if l1 and l2:
                val = carry + l1.val + l2.val
                carry = val // 10
                final = val % 10
                dummy.next = ListNode(final)
            if l1.next and not l2.next:
                l2.next = ListNode(0)
            if not l1.next and l2.next:
                l1.next = ListNode(0)
            l1 = l1.next
            l2 = l2.next
            dummy = dummy.next
        if carry: dummy.next = ListNode(carry)
        return res.next
            
        