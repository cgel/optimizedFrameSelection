import numpy as np
import h5py
import datetime
import operator
from functools import reduce
import os
import random
import math
import time
import cProfile as cP




TRAIN_RATIO = 0.9
SLICE_SIZE = 200


"""
==========PERFORMANCE EVALUATION==========
SPECS:
    Intel(R) Atom (TM) x5-Z8300 CPU @ 1.44GHz
    4.00Go Ram
    Windows 10 64bits
==========
PARAMETERS:
    SLICE_SIZE = 200
    TRAIN_RATIO = 0.9
==========
RESULTS:
    1000 uint8 elements of size [10, 10, 10, 10] and batches of 100
    db creation time:1.262
    batch creation time:6.325
    batch load time:6.064691066741943 for 100 batches
    raw load time:52.64411544799805 for 100 batches
    44.581 Mo on memory
--
    10000 uint8 elements of size [10, 10, 10, 10] and batches of 100
    db creation time:13.798
    batch creation time:70.114
    batch load time:6.173628807067871 for 100 batches
    raw load time:60.94627594947815 for 100 batches
    444.964 Mo on memory
--
    100000 uint8 elements of size [10, 10, 10, 10] and batches of 100
    db creation time:142.547
    batch creation time:1378.497
    batch load time:6.798546552658081 for 100 batches
    raw load time:66.31550431251526 for 100 batches
    4.448 Go on memory
"""





"""
!!!!!!!!ERRORS ENCOUNTERED!!!!!!!!



"""



"""
********TO DO********
    - Manage different size databases

    - New class datahandler, this class will become hdf5handler and inherit
        from datahandler class or datahandler will load hdf5 handler methods

    - slow to save and read!! add csv and other methods
    - overwrite batches when create batches is called multiple times
"""




class dataHandler():
        except AttributeError:
            print("WRONG DATA TYPE, EXPECTED NUMPY ARRAYS: CANCEL OPERATION")
            raise

        return dataArray, dataShape


    def getDataShape(self):
        with h5py.File(self.fileName, "r",  libver='latest') as f:
            return f["flatData/"].attrs["dataShape"]

    def initBuffer(self):
        """
        @@@@@@@@@@
        initBuffer
        @@@@@@@@@@

        description:
          Initialize the buffer to empty arrays.

        args:
          TBC

        returns:
          NA

        examples:
          dh.initBuffer()
        """
        self.buffer["data"] = np.zeros([self.sliceSize,self.dataLength])
        self.buffIndex = 0




    def load(self, index_or_list):
        """
        @@@@@@@@@@
        load
        @@@@@@@@@@

        description:
          loads a single or a list element from the hdf5 file.

        args:
          index_or_list : index of a single item or list.

        returns:
          data : the data requested. If a list was requested, the data will
                 be sorted. Will return false if the range is out of bound.
          index_or_list : returns the index first requested. this is useful
                          if the list has been sorted.

        examples:
          returnedData , _ = dh.load(20) #returns element at index 20.
          returnedData , _ = dh.loadRange([20,60,200]) #returns elements at
                                                       #index 20, 60, 200.
          returnedData , sortedList = dh.loadRange([60,200,20]) #same as above.
          if isinstance(returnedData,bool):
              print("index off bound")
        """


        if isinstance(index_or_list, int) or \
           isinstance(index_or_list, np.int32):
            pass
        else:
            try:
                index_or_list.sort()
            except AttributeError:
                print("ERROR: Input not supported")
                raise


        if self.fileType == "hdf5":
            #Open the file in read mode and returns the data
            with h5py.File(self.fileName, "r",  libver='latest') as f:
                try:
                    return f["flatData/data"][list(index_or_list)]
                except (KeyError, ValueError):
                    #invalid index in
                    print("INVALID INDEXES, LOAD CANCELLED")
                    return False
        else:
            print("fileType not configured")
            raise


    def loadBatch(self, batchType = "training"):
        """
        @@@@@@@@@@
        loadBatch
        @@@@@@@@@@

        description:
          returns batches of batchSize from the batch database. This will
          return a single batch from index 0 to batchQuantity at every call.


        args:
          batchType : The type of the current batch, There is two options:
                          -training
                          -test

        returns:
          batch : The training batch of batchSize of the test batch

        examples:
          dh.createBatch(trainList,"training")
          dh.createBatch(testList,"test")

          for i in range(100):
              batchtrain = dh.loadBatch("training")

          batchtest = dh.loadBatch("test")

        """
        try:
            #first pass or dataset modified
            with h5py.File(self.fileName, "r",  libver='latest') as f:
                if batchType == "training":
                    batch = f["batch/training/" + str(self.batchIndex)][:]
                    self.batchIndex += 1
                    try:
                        if self.batchIndex >= self.batchQuantity:
                            self.batchIndex = 0
                    except AttributeError:
                        self.batchQuantity = \
                                f["batch/training"].attrs["batchQuantity"]

                    return batch
                elif batchType == "test":
                    return f["batch/test/data"][:]

        except OSError:
                print("ERROR: File does not exists")
                raise


    def loadRange(self, start, end):
        """
        @@@@@@@@@@
        loadRange
        @@@@@@@@@@

        description:
          loads a range of elements element from the hdf5 file. If start is
          larger than end, the two elements will be swapped and the range
          returned anyway.

        args:
          start : First elements of the range
          end : Last element of the range

        returns:
          data : the data requested in the specified range. will return False
                 if the range is out of bound.

        examples:
          returnedData = dh.loadRange(20,60)
          returnedData = dh.loadRange(60,20) #same as 20,60
          returnedData = dh.LoadRange(400000:400002)
          if isinstance(returnedData,bool):
              print("range off bound")
        """
        #swap start is bigger than end
        if start>end:
            start, end = end, start

        if self.fileType == "hdf5":
            #Open the file in read mode and returns the data
            with h5py.File(self.fileName, "r",  libver='latest') as f:
                try:
                    data = f["flatData/data"][start:end]

                    if data.shape[0] < end - start:
                        print("INVALID RANGE, LOAD CANCELLED")
                        data = False
                except KeyError:
                    #invalid range
                    print("INVALID RANGE, LOAD CANCELLED")
                    data = False
        elif self.fileType == "csv":
            pass
        else:
            print("fileType not configured")
            return False
        return data


    def randList(self, batchSize):
        """
        @@@@@@@@@@
        randList
        @@@@@@@@@@

        description:
          Returns two lists containing random indexes for training and test.
          The training list will be split into batches of batcheSize.

        args:
          batchSize : Size of the batches for the training

        returns:
          trainingList : List containing the indexes for the training. This
                         list is separated in batches to be iterated
          testList : List containing the indexes for the test.

        examples:
          trainList, testList= dh.randList(100)
        """
        #check the size of the database
        with h5py.File(self.fileName, "r",  libver='latest') as f:
            qty = f["flatData/quantity"][0]
        #random list with unique labels
        temp = random.sample(range(0, qty),qty)
        #round up to next batchsize the train indexes
        endTrain = int(math.ceil(qty*TRAIN_RATIO / batchSize)) * batchSize
        #generate the two lists
        trainList, testList = temp[:endTrain], temp[endTrain:]
        #split the trainlist into batches
        trainList = np.array_split(trainList,endTrain/batchSize)
        return trainList, testList



    def saveData(self, dataType = "hdf5"):
        """
        @@@@@@@@@@
        saveData
        @@@@@@@@@@

        description:
          saves the buffer content to the hdf5 file. Will resize the dataset
          to the shape of the buffer.

        args:
          NA

        returns:
          NA

        examples:
          dh.saveData()
        """
        #if saveData is called with buffer not full
        if self.buffIndex < self.sliceSize:
            if self.buffIndex == 0:
                print("CANNOT SAVE: BUFFER EMPTY. SAVE CANCELLED")
                return
            resize = self.buffIndex
        else:
            resize = self.sliceSize

        if self.fileType == "hdf5":
            with h5py.File(self.fileName, "a",  libver='latest') as f:
                #for first write, create group
                if self.maxDataIndex == 0:
                    grp = f.create_group("flatData")
                    grp.attrs["dataShape"] = self.createIterableShape(self.dataShape)
                    grp.create_dataset("data",
                                       data=self.buffer["data"],
                                       compression="lzf",
                                       chunks=True,
                                       maxshape=(None, self.dataLength))
                    grp.create_dataset("quantity",
                                       data=[resize])
                #if first group created, resize and add new data
                else:
                    f["flatData/data"].resize(
                        f["flatData/data"].shape[0]+resize,axis=0)
                    f["flatData/data"][-resize:] = self.buffer["data"][:resize]
                    f["flatData/quantity"][0] = resize + \
                                                    f["flatData/quantity"][0]
                    self.maxDataIndex += resize
        else:
            print("fileType not configured")
            raise





    def setFilename(self, filename):
        """
        @@@@@@@@@@
        setFilename
        @@@@@@@@@@

        description:
          Set the filename to be accessed with the different operations

        args:
          filename : string with the name

        returns:
          NA

        examples:
          dh.setFilename("test.hdf5")
        """
        self.fileName = str(os.getcwd()) +"\\" + filename





#TEST SECTION
if __name__ == "__main__" and False:
    #test parameters
    DB_SIZE = 100000
    DATA_SHAPE = [28,28,10]
    BATCH_SIZE = 100
    BATCH_LOAD = 100

    print("{} elements of size {} and batches of {}".format(DB_SIZE,
                                                             DATA_SHAPE,
                                                             BATCH_SIZE))

    #datahandler object
    dh = dataHandler("hdf5")


    t = time.time()
    for _ in range(DB_SIZE):
        fake1=np.random.randint(255,size = [28,28], dtype= np.uint8)
        fake2=np.random.randint(255,size = [10], dtype= np.uint8)
#        fake2=np.random.randint(255,size = [1], dtype= np.uint8)
        dh.addData(fake1,fake2)
    print("db creation time:%.3f"%(time.time()-t))
    del fake1, fake2
    #save the elements in the buffer
    dh.saveData()


    #create random lists of batchSize
    trainList, testList= dh.randList(BATCH_SIZE)

    t = time.time()
    #create the batch datasets
    dh.createBatch(trainList,"training")
    dh.createBatch(testList,"test")
    print("batch creation time:%.3f"%(time.time()-t))

    #Load from the batches
    t = time.time()
    for i in range(BATCH_LOAD):
        batch1 = dh.loadBatch("training")
    print("batch load time:{} for {} batches".format((time.time()-t),BATCH_LOAD))

    #load from the raw data
    t = time.time()
    for i in range(BATCH_LOAD):
        batch2 = dh.load(trainList[0])
    print("raw load time:{} for {} batches".format((time.time()-t),BATCH_LOAD))


