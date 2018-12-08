import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from timeit import default_timer as timer
# needed to see images
from IPython.display import display, Image
import pickle
from sklearn.preprocessing import LabelBinarizer

# needed for plotting
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
current_palette = sns.color_palette()

# constants
numberOfLabels = 10
batchSize = 97
check_size = 97
feature_size = 193
n_hidden = 250
beta = 0.04
data = pickle.load(open('193_features.p', 'rb'))

datasetFrame = list(data['sample'])
datasetFrame = pd.DataFrame(datasetFrame)
data_cols = datasetFrame.columns
datasetFrame['label'] = data['label']
print('working dataframe\'s shape:', datasetFrame.shape)

# train test split
test_preds = {}
train = datasetFrame[0:6984]
test = datasetFrame[6984:]
LB = LabelBinarizer().fit(train['label'])
print(LB)
test_labels = LB.transform(test['label'])
print(test_labels)

del data, datasetFrame

def computeAccuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])

def setWeightVariable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.stack(initial)

def setBiasVariable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.stack(initial)

def testComputeAccuracy(session, test_data=test, during = True):
    test_data.reset_index(inplace=True, drop=True)
    epoch_pred = session.run(prediction, feed_dict={tf_data : test_data.loc[0:check_size-1,data_cols], keep_prob : 1.0})
    for i in range(check_size, test_data.shape[0], check_size):
        epoch_pred = np.concatenate([epoch_pred, session.run(prediction, 
                                    feed_dict={tf_data : test_data.loc[i:i+check_size-1,data_cols], keep_prob : 1.0})], axis=0)
    if during:
        print(computeAccuracy(epoch_pred, test_labels))
        return computeAccuracy(epoch_pred, test_labels)
    else:
        print(epoch_pred)
        return epoch_pred


acc_over_time = {}
def runRNNSession(numberOfEpochs, name, k_prob=1.0, mute=False, record=False):
    global train
    
    start = timer()
    with tf.Session(graph=graph) as session:
        if record:
            merged = tf.merge_all_summaries()  
            writer = tf.train.SummaryWriter("/tmp/tensorflowlogs", session.graph)
        #tf.initialize_all_variables().run()
        tf.global_variables_initializer().run()

        print("RNN is Initialized")
        accu = []
        
        for epoch in range(numberOfEpochs):
            
            # get batch
            trainingBatch = train.sample(batchSize)
            
            t_d = trainingBatch[data_cols]
            t_l = LB.transform(trainingBatch['label'])
            
            # make feed dict
            feed_dict = { tf_data : t_d, train_labels : t_l, keep_prob : k_prob}
            
            # run RNNModel on batch
            _, l, predictions = session.run([optimizer, loss, prediction], feed_dict=feed_dict)
            
            # mid RNNModel computeAccuracy checks 
            if (epoch % 1000 == 0) and not mute:
                print("\tMinibatch loss at epoch {}: {}".format(epoch, l))
                print("\tMinibatch computeAccuracy: {:.1f}".format(computeAccuracy(predictions, t_l)))
            if (epoch % 5000 == 0) and not mute:
                print("Test computeAccuracy: {:.1f}".format(testComputeAccuracy(session, during=True)))
            if (epoch % 1000 == 0) and not mute:
                accu.append(tuple([epoch, testComputeAccuracy(session, during=True)]))
                
        # record computeAccuracy and predictions
        test_preds[name] = testComputeAccuracy(session, during=False)
        print("Final Test computeAccuracy: {:.1f}".format(computeAccuracy(test_preds[name], test_labels)))
        end = timer()
        test_preds[name] = test_preds[name].ravel()
        acc_over_time[name] = accu
        print("time taken: {0} minutes {1:.1f} seconds".format((end - start)//60, (end - start)%60))

graph = tf.Graph()
with graph.as_default():
    # placeholders
    tf_data = tf.placeholder(tf.float32, shape=[None, feature_size])
    train_labels = tf.placeholder(tf.float32, shape=[None, numberOfLabels])
    keep_prob = tf.placeholder(tf.float32)
    
    # weights and biases
    layer1Weights = setWeightVariable([feature_size*n_hidden, numberOfLabels])
    layer1_biases = setBiasVariable([numberOfLabels])
    
    # RNNModel
    def RNNModel(data, proba=1.0):
        # Init RNN cell # tensorflow seems to take care of not reinitializing each time RNNModel is called
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)
        # run rnn cell
        layer1, _istate = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
        # reshape for output layer.
        layer1 = tf.reshape(layer1, shape=[batchSize, feature_size*n_hidden])
        layer2 = tf.nn.dropout(layer1, proba)
        return tf.matmul(layer2, layer1Weights) + layer1_biases

    # Training computation.
    logits = RNNModel(tf.expand_dims(tf_data, -1), keep_prob)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=train_labels) +
                         beta*tf.nn.l2_loss(layer1Weights) +
                         beta*tf.nn.l2_loss(layer1_biases))

    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

    prediction = tf.nn.softmax(logits) 
    print('Basic RNN RNNModel made')

runRNNSession(5000, 'RNN', .2)