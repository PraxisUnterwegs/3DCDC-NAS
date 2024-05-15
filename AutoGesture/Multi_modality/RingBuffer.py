class RingBuffer:
    def __init__(self, capacity, default=None):
        self.capacity = capacity
        self.buffer = [default] * capacity
        self.size = 0
        self.head = 0  # Pointer to the next element to be overwritten
        self.tail = 0  # Pointer to the next available position

    def is_empty(self):
        return self.size == 0

    def is_full(self):
        return self.size == self.capacity

    def push(self, item):
        if self.is_full():
            # Overwrite the oldest element
            self.buffer[self.head] = item
            self.head = (self.head + 1) % self.capacity
        else:
            self.buffer[self.tail] = item
            self.tail = (self.tail + 1) % self.capacity
            self.size += 1

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from an empty buffer")
        else:
            item = self.buffer[self.head]
            self.head = (self.head + 1) % self.capacity
            self.size -= 1
            return item
    def get(self):
        return self.buffer

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"RingBuffer({self.buffer})"