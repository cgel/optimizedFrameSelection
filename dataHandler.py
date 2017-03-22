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
    - Manage the deletion of the object (to empty the buffer before close)

    - Manage int as inputs and others.
"""




class dataHandler():
    """
    ##############
    dataHandler
    ##############
    description:
      Class to handle data in and out of an external hdf5 file with h5py.
      Can be used to generate a dataset or read from an existing one.

    example:
        #test parameters
        DB_SIZE = 10000
        BATCH_SIZE = 100
        BATCH_LOAD = 100


        #datahandler object
        dh = dataHandler()

        #fake data
        fake1=np.random.randint(255,size = [7056])
        fake2=np.random.randint(255,size = [4,4])
        fake3=np.random.randint(255,size = [512])
        fake4=np.random.randint(255,size = [28,28,28])

        #save loop
        for _ in range(DB_SIZE):
            dh.addData(fake1,fake2,fake3,fake4)
        del fake1, fake2, fake3, fake4

        #Empty the buffer and save it
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
        """


    def __init__(self, fileType = "hdf5"):

        if not isinstance(fileType, str):
            print("ERROR : fileType must be a string")
            raise

        #Max index in the dataset
        self.maxDataIndex = 0

        #Size of the save slice in memory
        self.sliceSize = SLICE_SIZE

        #length of the flatten array with the data
        self.dataLength = 0
        self.dataShape = []

        #Buffer used for single add in memory. This is to avoid to
        #write in the hdf5 file for every new item (painfully slow)
        self.buffer = {}

        #index for batch reading
        self.batchIndex = 0

        #filetype of the datahandler (to remove)
        self.fileType = fileType

        #filename with timestamp.
        self.fileName = str(os.getcwd()) +"\\dh_output_{}.{}".format(
            datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
            self.fileType)





    def addData(self, *data):
        """
        @@@@@@@@@@
        addData
        @@@@@@@@@@

        description:
          Add data to the dataHandler object. When the buffer is full, the
          function will make a call to saveData to save the buffer in the
          external hdf5 file.

        args:
          *data : all the data that will be saved in the hdf5 file. Must be
                  numpy arrays. Can be any dimensions except 0.

        returns:
          NA

        examples:
          for _ in range(100000):
            dh.addData(fakeData1, fakeData2, fakeData3)
        """
        #flatten the data to a 1D array
        flat, shape = self.flattenData(*data)

        #datatype error occured
        if isinstance(flat, bool):
            return

        #first add to the file, note the shape of the array
        if self.dataLength == 0:
            self.dataLength = sum(shape)

            self.dataShape = self.createIterableShape(shape)
            self.initBuffer()
#        else:
            #For now if the shape is different, cancel operations
            #Possible add to the future: create new datasets when different
            #input sizes.
#            if shape != self.dataShape:
#                print("DIFFERENT DATA SHAPE: ABORT WRITE")
#                raise



        #if buffer full, save to hdf5 file
        if self.buffIndex >= self.sliceSize:
            self.saveData()
            self.initBuffer()
        #assign data to the buffer
        self.buffer["data"][self.buffIndex] = flat

        self.buffIndex += 1




    def createBatch(self, randomList, batchType = "training"):
        """
        @@@@@@@@@@
        createBatch
        @@@@@@@@@@

        description:
          Creates batches in the dataset for fast reads. This operation
          takes time but allows to speed up reading up to 100 times. The
          batches will be in a group batch/train(or test).

        args:
          randomList : Random list of indexes from the original database. This
                       list must be sliced into batches.
          batchType : The type of the current batch, There is two options:
                          -training
                          -test

        returns:
          NA

        examples:
          dh = dataHandler("hdf5")
          trainList, testList= dh.randList(100)

          dh.createBatch(trainList,"training")
          dh.createBatch(testList,"test")
        """
        try:
            with h5py.File(self.fileName, "a",  libver='latest') as f:
                #create training dataset
                if batchType == "training":
                    #group and attributes
                    grp = f.create_group("batch/training")
                    grp.attrs["batchQuantity"] = len(randomList)
                    grp.attrs["batchSize"] = len(randomList[0])
                    f["batch/training/list"] = randomList

                    startTime = time.time()

                    #Create batches
                    for i, batchList in enumerate(randomList):
                        grp.create_dataset(str(i),
                                           data=self.load(batchList),
                                           chunks=(len(batchList),
                                                   self.dataLength),
                                           compression="lzf",
                                           maxshape=(len(batchList),
                                                     self.dataLength))
                        if i == 0:
                            print("Creating train batch, time estimate=%.2fsec"
                            %((time.time()-startTime)*len(randomList)))

                #create test dataset
                elif batchType == "test":
                    print("Create test batch")
                    grp = f.create_group("batch/test")
                    f["batch/test/list"] = randomList
                    grp.create_dataset("data",
                                       data=self.load(randomList),
                                       compression="lzf",
                                       chunks=True,
                                       maxshape=(len(randomList),
                                                 self.dataLength))
                else:
                    raise
        except OSError:
                print("ERROR: File does not exists")
                raise


    def createIterableShape(self,shape):
            b = []
            for index, data in enumerate(shape):
                if index == 0:
                    b.append([0,data])
                else:
                    b.append([b[index-1][1],b[index-1][1]+data])
            return b


    def flattenData(self,*data):
        """
        @@@@@@@@@@
        flattenData
        @@@@@@@@@@

        description:
          Receives the data as separated elements in the function call and
          returns the same information as a single 1D numpy array.

        args:
          *data : data to be flatten to an array.

        returns:
          dataArray : The array with the data.
          dataShape : the shape of the new array. Can be used to indicate the
                      separations in the new array

        examples:
          d1 = np.ones([153])
          d2 = np.zeros([4,4])
          d3 = np.ones([1])
          d4 = np.zeros([1,6,6,1])
          flat,shape = dh.flattenData(d1,d2,d3,d4)
              #flat = flat array of 206 -> [d1,d2,d3,d4]
              #shape = [153, 16, 1, 36]
          flat,shape = dh.flattenData(d1,d2,d3,d4,[1,2,3])
              #flat & shape = False: datatype not supported
          flat,shape = dh.flattenData(d1,d2,d3,d4,(1,2,3))
              #flat & shape = False: datatype not supported
          flat,shape = dh.flattenData(d1,d2,d3,d4,1)
              #flat & shape = False: datatype not supported
        """
        if len(data) == 0:
            print("NO DATA")
            return False, False

        #variables to contain the flatten data and shape as a list
        dataArray = np.zeros(0)
        dataShape = []

        try:
            for element in data:
                #already 1d array
                if len(element.shape) == 1:
                    dataArray = np.append(dataArray, element)
                    dataShape += [element.shape[0]]
                #reshape to a 1d array
                else:
                    new_shape = reduce(operator.mul,element.shape,1)
                    dataShape += [new_shape]
                    dataArray = np.append(dataArray,
                                          np.reshape(element,
                                                     new_shape))
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
                    grp.attrs["dataShape"] = self.dataShape
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
if __name__ == "__main__":
    #test parameters
    DB_SIZE = 20000
    DATA_SHAPE = [28,28,10]
    BATCH_SIZE = 100
    BATCH_LOAD = 100

    print("{} elements of size {} and batches of {}".format(DB_SIZE,
                                                             DATA_SHAPE,
                                                             BATCH_SIZE))

    #datahandler object
    dh = dataHandler("hdf5")

#[(1, 84, 84), (1, 4), (1,), (1,), (1,), (1, 3136), (1, 512)]
    t = time.time()
    for _ in range(DB_SIZE):
        fake1=np.random.randint(255,size = [1,5,5], dtype= np.uint8)
        fake2=np.random.randint(255,size = [1,4], dtype= np.uint8)
        fake3=np.array([1])
        fake4=np.random.randint(255,size = [1,400], dtype= np.uint8)
        dh.addData(fake1,fake2, fake3, fake4)
    print("db creation time:%.3f"%(time.time()-t))
    del fake1, fake2, fake3, fake4
    #save the elements in the buffer
    dh.saveData()


    #create random lists of batchSize
    trainList, testList= dh.randList(BATCH_SIZE)

    if len(testList) == 0:
        print("Not enough data to create batches")
        raise

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


