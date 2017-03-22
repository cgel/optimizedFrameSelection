import tensorflow as tf
import time
import imitParam
import dataHandler as dh
import numpy as np
from sklearn.neighbors import LSHForest




class imitationLearner():
    def __init__(self, session, param = imitParam.imitParam):
        #learner session
        self.sess = session
        #parameters of the learner
        self.param = param
        #placeholders
        self.input_ph = tf.placeholder(
                tf.float32, self.param.input_shape, name="inputData_ph")
        self.output_ph = tf.placeholder(
                tf.float32, self.param.output_shape, name="output_ph")
        self.label_ph = tf.placeholder(
                tf.float32, self.param.output_shape, name="label_ph")
        self.keep_prob = tf.placeholder(
                tf.float32, name="dropout_ph")

        #Creates the learner and every operation necessary
        self.learner = self.createLearner()
        self.loss_op, _ = self.loss()
        self.train_op = self.training()
        self.eval_op = self.evaluation()

        #datahandler for data manipulation
        self.dh = dh.dataHandler()


    def addConvLayer(self,inputs, shape):
        return tf.layers.conv2d(inputs=inputs,
                                filters=shape[0],
                                kernel_size=[shape[1],\
                                            shape[2]],
                                padding="same",
                                strides=shape[3],
                                activation=tf.nn.relu)

    def addDenseLayer(self,inputs, shape):
        return tf.layers.dense(inputs=inputs,
                               units=shape.pop(),
                               activation=tf.nn.relu)


    def createLearner(self, normalize = False):

        if normalize:
            input_norm = tf.div(self.input_ph, 255., name="normalized_input")
        else:
            input_norm = self.input_ph


        #Input reshape
        input_reshape_norm = tf.reshape(input_norm, [-1,28,28,1])#A mettre en parametres


        #create all conv layers
        convLayers = []
        for index, shape in enumerate(self.param.Conv_param):

            if index ==0:
                #first conv layer
                convLayers.append(self.addConvLayer(input_reshape_norm,shape))
            else:
                convLayers.append(self.addConvLayer(convLayers[index-1],shape))

        #index of the last conv layer
        lastConvLayerIndex = len(self.param.Conv_param)-1

        #conv to tense Layer conversion
        outConv = convLayers[lastConvLayerIndex]
        outShape = outConv.get_shape().as_list()
        lastConvFlat = tf.reshape(outConv,
                                [-1, outShape[1]*outShape[2]*outShape[3]])

        #create the dense layers
        denseLayers = []
        for index, shape in enumerate(self.param.fully_connected_layer):
            if index == 0:
                denseLayers.append(self.addDenseLayer(lastConvFlat,
                                                      shape))
            else:
                denseLayers.append(self.addDenseLayer(denseLayers[index-1],
                                                      shape))
        #index of the last
        lastDenseLayerIndex = len(self.param.fully_connected_layer)-1

        #drouput layer
        drop = tf.nn.dropout(denseLayers[lastDenseLayerIndex], self.keep_prob)

        #Output Layer
        self.output_ph = tf.layers.dense(inputs=drop,
                                         units=self.param.output_shape[1])





    def loss(self):
        loss_op = tf.nn.softmax_cross_entropy_with_logits(
                                                      labels=self.label_ph,
                                                      logits=self.output_ph)
        return loss_op, tf.reduce_mean(loss_op)




    def training(self):
        # Add a scalar summary for the snapshot loss.
        tf.summary.scalar('loss', tf.reduce_mean(self.loss_op))
        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer(self.param.learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        #(and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(self.loss_op, global_step=global_step)
        return train_op



    def evaluation(self):
        correct_prediction = tf.equal(tf.argmax(self.label_ph,1),
                                      tf.argmax(self.output_ph,1))
        #Cast to floating point numbers and then take the mean
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



    def run(self):
        #Loads the MNIST dataset
        self.dh.setFilename("mnist.hdf5")
        dhGen = dh.dataHandler()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        #start the session
        self.sess.run(init)

        #load the test batch
        print(" ".join(["loading",self.dh.fileName]))
        testBatch = self.dh.loadBatch("test")

        #separate data
        dataShape = self.dh.getDataShape()
        testData = testBatch[:,dataShape[0,0]:dataShape[0,1]]
        testLabel = testBatch[:,dataShape[1,0]:dataShape[1,1]]

        #Training loop
        print("Begin training")
        start_time = time.time()
        for step in range(self.param.max_steps):


            batch = self.dh.loadBatch()

            feed_dict = {self.input_ph:batch[:,dataShape[0,0]:dataShape[0,1]],
                         self.label_ph:batch[:,dataShape[1,0]:dataShape[1,1]],
                         self.keep_prob: 0.5}

            # Run one step of the model.
            _, loss = self.sess.run([self.train_op, self.loss_op],
                                          feed_dict=feed_dict)

            if step >= 20000:
                self.param.learning_rate = 0
                for data in batch[:,dataShape[0,0]:dataShape[0,1]]:
                    dhGen.addData(data, loss)

            #Show loss of network
            if step % 100 == 0:
              duration = time.time() - start_time
              print('Step {0}: loss = {1:.3f} ({2:.3f} sec)'.format(step, np.mean(loss), duration))
              start_time = time.time()


            #Evaluate the model periodically.
            if (step + 1) % 2500 == 0 or (step + 1) == self.param.max_steps:
                print("test accuracy %g"%self.sess.run(self.eval_op,
                                                  feed_dict={
                                            self.input_ph: testData,
                                            self.label_ph: testLabel,
                                            self.keep_prob: 1.0}))
                start_time = time.time()
        dhGen.saveData()


if __name__ == "__main__":
    sess = tf.Session()
    imit = imitationLearner(sess)
    imit.run()
