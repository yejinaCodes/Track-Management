#object list
class frame():
    #add object list
    def __init__ (self):
        self.objects_lists=[]
        self.max_length = 5
    def add(self,objects):
        self.list.append(objects)
        if len(self.objects_lists) > self.max_length:
            print('too many objects detected. More than 5')
