# Put in basic everything doesn't involve torch

class Stack:
    def __init__(self, name='', parent=None):
        self.parent = parent
        self.keys = {}
        self.subs = {}

        self.name = name
        
    def enter(self, name):
        if name in self.subs:
            return self.subs[name]
        else:
            sub = Stack(name, self)
            self.subs[name] = sub
            return sub 
    
    def exit(self):
        return self.parent

    def set(self, key, val):
        self.keys[key] = val

    def get(self, key, rec = 0):
        if rec > 10:
            return None

        if key not in self.keys:
            if self.parent != None:
                return self.parent.get(key, rec+1)
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