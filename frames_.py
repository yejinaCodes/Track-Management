#frame list
class frames():
    def __init__ (self):
        self.list=[]
        self.max_length = 90
    def add(self, frame):
        self.list.append(frame)
        if len(self.list) > self.max_length:
            self.list = self.list[31:]




    