import glob
import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
%matplotlib inline
plt.style.use('ggplot')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 13

numberOfFrames = 41
bands = 60
featureSize = 2460 
num_labels = 10
num_channels = 2
batch_size = 64
kernel_size = 30
depth = 64
num_hidden = 1050
learning_rate = 0.0001
trainingIterations = 20000


def setWindowSize(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)

def extractFeaturesfromData(parentDirectory,subDirectories,fileExtension="*.wav",bands = 60, numberOfFrames = 41):
    window_size = 512 * (numberOfFrames - 1)
    log_specgrams = []
    labels = []
    for l, sub_dir in enumerate(subDirectories):
        for fn in glob.glob(os.path.join(parentDirectory, sub_dir, fileExtension)):
            sound_clip,s = librosa.load(fn)
            label = fn.split('/')[2].split('-')[1]
            for (start,end) in setWindowSize(sound_clip,window_size):
                if(len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]
                    melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
                    logspec = librosa.amplitude_to_db(melspec)
                    logspec = logspec.T.flatten()[:, np.newaxis].T
                    log_specgrams.append(logspec)
                    labels.append(label)
            
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,numberOfFrames,1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    
    return np.array(features), np.array(labels,dtype = np.int)

def oneHotEncodeData(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    oneHotEncodeData = np.zeros((n_labels,n_unique_labels))
    oneHotEncodeData[np.arange(n_labels), labels] = 1
    return oneHotEncodeData

def loadDataFiles():
    parentDirectory = 'Sound'
    subDirectories= ['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10']
    features,labels = extractFeaturesfromData(parentDirectory,subDirectories)
    labels = oneHotEncodeData(labels)

def setWeightsVariable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def setBiasVariable(shape):
    initial = tf.constant(1.0, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,2,2,1], padding='SAME')

def applyConvConfiguration(x,kernel_size,num_channels,depth):
    weights = setWeightsVariable([kernel_size, kernel_size, num_channels, depth])
    biases = setBiasVariable([depth])
    return tf.nn.relu(tf.add(conv2d(x, weights),biases))

def applyMaxPoolingLayer(x,kernel_size,stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1], 
                          strides=[1, stride_size, stride_size, 1], padding='SAME')

loadDataFiles();
rnd_indices = np.random.rand(len(labels)) < 0.70

train_x = features[rnd_indices]
train_y = labels[rnd_indices]
test_x = features[~rnd_indices]
test_y = labels[~rnd_indices]

X = tf.placeholder(tf.float32, shape=[None,bands,numberOfFrames,num_channels])
Y = tf.placeholder(tf.float32, shape=[None,num_labels])

cov = applyConvConfiguration(X,kernel_size,num_channels,depth)

shape = cov.get_shape().as_list()
cov_flat = tf.reshape(cov, [-1, shape[1] * shape[2] * shape[3]])

f_weights = setWeightsVariable([shape[1] * shape[2] * depth, num_hidden])
f_biases = setBiasVariable([num_hidden])
f = tf.nn.sigmoid(tf.add(tf.matmul(cov_flat, f_weights),f_biases))

out_weights = setWeightsVariable([num_hidden, num_labels])
out_biases = setBiasVariable([num_labels])
y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)

X = tf.placeholder(tf.float32, shape=[None,bands,numberOfFrames,num_channels])
Y = tf.placeholder(tf.float32, shape=[None,num_labels])

cov = applyConvConfiguration(X,kernel_size,num_channels,depth)

shape = cov.get_shape().as_list()
cov_flat = tf.reshape(cov, [-1, shape[1] * shape[2] * shape[3]])

f_weights = setWeightsVariable([shape[1] * shape[2] * depth, num_hidden])
f_biases = setBiasVariable([num_hidden])
f = tf.nn.sigmoid(tf.add(tf.matmul(cov_flat, f_weights),f_biases))

out_weights = setWeightsVariable([num_hidden, num_labels])
out_biases = setBiasVariable([num_labels])
y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)

cross_entropy = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
costHistory = np.empty(shape=[1],dtype=float)
with tf.Session() as session:
    tf.global_variables_initializer().run()

    for iteration in range(trainingIterations):    
        offset = (iteration * batch_size) % (train_y.shape[0] - batch_size)
        batch_x = train_x[offset:(offset + batch_size), :, :, :]
        batch_y = train_y[offset:(offset + batch_size), :]
        
        _, c = session.run([optimizer, cross_entropy],feed_dict={X: batch_x, Y : batch_y})
        costHistory = np.append(costHistory,c)
    
    print('Test accuracy of the CNN model is ',round(session.run(accuracy, feed_dict={X: test_x, Y: test_y}) , 3))
    fig = plt.figure(figsize=(15,10))
    plt.plot(costHistory)
    plt.axis([0,trainingIterations,0,np.max(costHistory)])
    plt.show()