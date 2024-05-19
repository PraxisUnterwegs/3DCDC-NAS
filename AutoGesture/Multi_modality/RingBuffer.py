import numpy as np
import time

class RingBuffer:
    def __init__(self, capacity, default=None):
        self.capacity = capacity  # max number of elements in the buffer
        self.buffer = [default] * capacity  # initialize the buffer with default values(filling)
        self.size = 0  # initialize the variable size, which store the number of elements in the buffer
        self.head = 0  # initialize the buffer head index(it will change with new elements inserted)
        self.tail = 0  # initialize the buffer tail index(it will change with new elements inserted)
        self.last_get_time = time.time()

    # check whether the buffer is empty
    def is_empty(self):  
        return self.size == 0

    # check whether the buffer is full
    def is_full(self):
        return self.size == self.capacity

    # add a new element into the buffer, if the buffer is full, it will overwriter the oldest element(head)
    # and move the head index forward one step
    # if the buffer still has space, it will add a new element to tail of the buffer, and move the tail index forward one step
    # head index will increase until the buffer is full, tail index will increase if the buffer still has space
    # head对应读索引，tail对应写索引 https://zhuanlan.zhihu.com/p/534098236 
    def push(self, item):
        if self.is_full():
            # Overwrite the oldest element
            # head index forward
            self.buffer[self.head] = item
            self.head = (self.head + 1) % self.capacity
        else:  # if buffer still has space
            # tail index forward
            self.buffer[self.tail] = item
            self.tail = (self.tail + 1) % self.capacity
            self.size += 1

    # Removes and returns the oldest element from the buffer
    def pop(self):
        if self.is_empty():  # if the buffer is empty, and u call pop(), then it will actively throw errors
            raise IndexError("pop from an empty buffer")
        else:
            item = self.buffer[self.head]  # 返回的目标是读索引（head）, head 指向 oldest element
            self.head = (self.head + 1) % self.capacity  # 如果读出去了，head forward to 下一个 oldest element
            self.size -= 1
            return item
        
    def get(self):  # 返回buffer里的所有元素
        # if time.time() - self.last_get_time >= 2:
        #     self.last_get_time = time.time()
        #     if self.is_full():
        #         output_tensor = self.buffer
        #         self.clear()
        #         return output_tensor
        return self.buffer
    
    def clear(self):
        # 清空缓冲区
        self.buffer = [None] * self.capacity
        self.size = 0
        self.head = 0
        self.tail = 0
        

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"RingBuffer({self.buffer})"