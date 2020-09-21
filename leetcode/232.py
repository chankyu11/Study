class MyQueue:
    
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.queue = deque()
        

    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        self.queue.append(x)

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if len(self.queue) >0:
            return self.queue.popleft()
        else:
            return None

    def peek(self) -> int:
        """
        Get the front element.
        """
        if len(self.queue) > 0:
            return self.queue[0]
        else:
            return None
    

    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        return len(self.queue) == 0


# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()