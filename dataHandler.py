import pickle
import os

class dataHandler():
    def __init__(self):
        self.data = {"frame"        :{}, 
                     "qVal"         :{}, 
                     "action"       :{}, 
                     "reward"       :{},
                     "isTerminal"   :{},
                     "internal1"    :{},
                     "internal2"    :{}}
        self.qty    = 0
        self.pklQty = 0
        
        self.dir = os.path.join(os.path.dirname(__file__), '/dataset/')
        
        
    def emptyHandler(self):
        self.data.clear()
        self.data = {"frame"        :{}, 
                     "qVal"         :{}, 
                     "action"       :{}, 
                     "reward"       :{},
                     "isTerminal"   :{},
                     "internal1"    :{},
                     "internal2"    :{}}
        self.qty = 0
        
    def addData(self,
                frame, 
                expectedQ, 
                action, 
                reward, 
                isTerminal, 
                internal1, 
                internal2):
        
        self.data["frame"][self.qty]      = frame
        self.data["qVal"][self.qty]       = expectedQ
        self.data["action"][self.qty]     = action
        self.data["reward"][self.qty]     = reward
        self.data["isTerminal"][self.qty] = isTerminal
        self.data["internal1"][self.qty]  = internal1
        self.data["internal2"][self.qty]  = internal2
        self.qty += 1
        
        if self.qty > 1000000:
            self.pickleSave()
        
    
    def pickleSave(self):
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            
        with open(self.dir + str(self.pklQty) + '.pickle', 'wb') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.emptyHandler()
            self.pklQty += 1
            
    def pickleLoad(self,datasetID):
        with open(self.dir + str(datasetID) + '.pickle', 'rb') as handle:
            self.data   = pickle.load(handle)
            self.qty    = len(self.data["isTerminal"]) 
            self.pklQty = datasetID
