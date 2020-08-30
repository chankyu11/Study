# # Definition for singly-linked list.
# # class ListNode:
# #     def __init__(self, val=0, next=None):
# #         self.val = val
# #         self.next = next
# class Solution:
#     def reverseList(self, head: ListNode) -> ListNode:
#         node, prev = head, None
        
#         while node:
#             next, node.next = node.next, prev
#             prev, node = node, next
            
#         return prev
        

    

# # 2번

# # Definition for singly-linked list.
# # class ListNode:
# #     def __init__(self, val=0, next=None):
# #         self.val = val
# #         self.next = next
# class Solution:
#     def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        
#         root = head = ListNode(0)
        
#         #9999 + 1
#         carry = 0
#         while L1 or L2 or carry:
#             mysum = 0
#             if L1:
#                 mysum += L1.val
#                 L1 = L1.next
#             if L2:
#                 mysum += L2.val
#                 L2 = L2.next
#             carry, val = divmod(mysum + carry, 10)
#             head.next = ListNode(val)
#             head = head.next
            
            
    
# # 20

a = ['ebcabc']
b = a.split()
print(b)
# print(sorted(b))