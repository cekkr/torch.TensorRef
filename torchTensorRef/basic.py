# Put in basic everything doesn't involve torch

class Stack:
    def __init__(self, name='', parent=None):
        self.parent = parent
        self.keys = {}

        self.name = name
        
    def enter(self, name):
        if name in self.keys:
            return self.keys[name]
        else:
            sub = Stack(name, self)
            self.keys[name] = sub
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
    
    def getFullName(self):
        name = self.name
        if self.parent is not None:
            fname = self.parent.getFullName()
            if len(fname) > 0:
                name = self.parent.getFullName() + '>>' + name
        return name