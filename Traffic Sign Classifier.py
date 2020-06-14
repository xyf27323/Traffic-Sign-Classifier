# Load pickled data
import pickle
import random
import numpy as np
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import PIL
import matplotlib.gridspec as gridspec
# TODO: Fill this in based on where you saved the training and testing data

training_file = "train.p"
valid_file = "valid.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
with open(valid_file,mode='rb') as f:
    valid = pickle.load(f)
X_train, y_train = train['features'], train['labels']
X_validation,y_validation =valid['features'],valid['labels']  
X_test, y_test = test['features'], test['labels']


# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: Number of testing examples.
n_validation = X_validation.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = np.max(y_train)+1

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Number of validation examples =", n_validation)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#展示图片
#sample_per_class = 3
#
#fig = plt.figure(figsize=(sample_per_class, n_classes))
#for i in range(n_classes):
#    class_i = y_train==i
#    x_class_i = X_train[class_i, ]
#    for j in range(sample_per_class):               
#        plt.subplot(n_classes,sample_per_class, i*sample_per_class+j+1)
#        plt.imshow(random.choice(x_class_i))
#        plt.axis('off')
#plt.savefig("train_set.png")
#plt.show

#Pre-process the Data Set (normalization, grayscale, etc.)
def CovToGray(data):    
    for index in range(0,data.shape[0]):
        for channel in range(0,2):
            data[index,:,:,channel]= cv2.cvtColor(data[index], cv2.COLOR_BGR2GRAY)
    return data[:,:,:,0:1]

X_train = X_train.astype('float32')
X_validation = X_validation.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255 - 0.5
X_validation = X_validation / 255 - 0.5
X_test = X_test / 255 - 0.5

X_train = CovToGray(X_train)
X_validation = CovToGray(X_validation)
X_test = CovToGray(X_test)


#Model Architecture
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
from tensorflow.contrib.layers import flatten
def LeNet(x):    
    # Hyperparameters 参数用于tf.truncated_normal，随机定义每个层的权重和偏差的变量
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma)) #a little change
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)
    
    #Add Dropout Layer
    conv1 = tf.nn.dropout(conv1, keep_prob)
    
    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
  
    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)
    
    #Add Dropout Layer
    conv2 = tf.nn.dropout(conv2,keep_prob)
    
    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

#Train, Validate and Test the Model

EPOCHS = 50
BATCH_SIZE = 64
X_train, y_train = shuffle(X_train, y_train)
X_validation,y_validation = shuffle(X_validation,y_validation)
X_test, y_test = shuffle(X_test, y_test)

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
keep_prob = tf.placeholder(tf.float32)

#TRAINING
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

#MODEL EVALUATION
X_train, y_train = shuffle(X_train, y_train)
X_validation,y_validation = shuffle(X_validation,y_validation)
X_test, y_test = shuffle(X_test, y_test)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))  #tf.argmax(input,axis)axis=1时比较每一行的元素，
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  #接上 将每一行最大元素所在的索引记录下来，最后输出每一行最大元素所在的索引数组
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

'''#Training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        Train_accuracy = evaluate(X_train, y_train)

        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Train Accuracy = {:.3f}".format(Train_accuracy))

        print()
        
    saver.save(sess, 'C:/Users/DELL/Desktop/Traffic-Sign-Classifier/Traffic Sign Classifier myself/model/model.ckpt')
    print("Model saved")
'''


##EVALUATION

with tf.Session() as sess:
#    saver = tf.train.import_meta_graph('C:/Users/DELL/Desktop/Traffic-Sign-Classifier/Traffic Sign Classifier myself/model/model.ckpt.meta') # 加载模型结构
    saver.restore(sess, tf.train.latest_checkpoint('C:/Users/DELL/Desktop/Traffic-Sign-Classifier/Traffic Sign Classifier myself/model'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


#Test a Model on New Images
from os import walk,path

files = []
image_dir = "new_image"
for dirpath,dirnames,filenames in walk(image_dir):
    files.extend(filenames)
#print(files)
image_data = np.zeros(shape=[len(files),32,32,4])
X_newtest = np.zeros(shape=[len(files),32,32,3])
y_newtest = np.zeros(shape=[len(files)])
   
for index,file in enumerate(files):
    image_data[index]= mpimg.imread(image_dir+"/"+file)
    label,extend= path.splitext(file)
    y_newtest[index] = int(label)

X_newtest=image_data[:,:,:,0:3]
X_newtest = X_newtest.astype('float32')


w=len(files)
fig = plt.figure(figsize=(w,1))
gs=gridspec.GridSpec(1,w,wspace=0.0,hspace=0.0)

for i in range (w):
    ax = plt.subplot(gs[0,i])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)

    ax.imshow(X_newtest[i])
plt.savefig("new_image.png")
plt.show()

#Predict the Sign Type for Each Image
X_newtest=CovToGray(X_newtest)

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('C:/Users/DELL/Desktop/Traffic-Sign-Classifier/Traffic Sign Classifier myself/model'))

    test_accuracy = evaluate(X_newtest, y_newtest)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


        