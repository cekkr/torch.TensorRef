# Put in basic everything doesn't involve torch

class Stack:
    def __init__(self, parent=None):
        self.parent = parent
        self.keys = {}
        
    def enter(self):
        sub = Stack(self)
        return sub 
    
    def exit(self):
        return self.parent

    def set(self, key, val):
        self.keys[key] = val

    def get(self, key):
        if key not in self.keys:
            if self.parent != None:
                return self.parent.get(key)
            else:
                return None 
        return self.keys[key]