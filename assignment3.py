import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D

# define some functions

def LoadBatch(dataset_name):
    dataset = sio.loadmat(dataset_name)
    data = np.float32(np.array(dataset['data'])) / np.max(np.float32(np.array(dataset['data'])))
    labels_1 = np.array(dataset['labels'])
    label_no_onehot = []
    for j in range(np.size(labels_1)):
        label_no_onehot.append(labels_1[j][0])
    label_no_onehot = np.array(label_no_onehot)
    data_length = np.size(labels_1)
    label_max = 10
    label = np.zeros([np.size(labels_1),label_max])
    for i in range(np.size(labels_1)):
        label[i][labels_1[i]] = 1
    data = np.transpose(data) # 3072 * 10000
    label = np.transpose(label) # 10000 * 10
    return data, label, data_length, label_no_onehot


def initialization(W_b_dimension):
    list_W = []
    list_b = []
    list_W.append(np.random.normal(0,0.001,[3072,W_b_dimension[0]]))
    list_b.append(np.zeros(W_b_dimension[0]))
    for i in range(1,np.size(W_b_dimension)):
        list_W.append(np.random.normal(0,0.001,[W_b_dimension[i],W_b_dimension[i-1]]))
        list_b.append(np.zeros(W_b_dimension[i]))
    return list_W, list_b

def ComputeCost(label, lam, batch_size, P, W1, W2):
    Y = label
    loss = -(1.0 / batch_size) * np.sum(Y * np.log(P))
    J = loss + lam * (np.sum(np.power(W1, 2)) + np.sum(np.power(W2, 2)))
    #J = loss+lam*np.sum(W**2)
    return J, loss

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def Compute_S(W, b, input_data):
    s = np.dot(W, input_data) + b
    return s

def Compute_h(s):       #ReLU
    h = np.maximum(0,s)
    return h


def EvaluationClassifier(s):
    P = softmax(s)
    return P

def sigmoid(s):
    h = 1/(1+np.exp(-s))
    return h


def ComputeGradients(W1, W2, b1, b2, P, input_data, input_label, lam, batch_size, m, h):

    # input_data with 3072 * Batch_size
    # input_label with 10 * Batch_size
    # g = np.mean(P - input_label,1)
    g = P - input_label
    grad_b2= np.mean(g,1)

    grad_W2 = np.dot(g,h.T)/batch_size + 2 * lam * W2
    g = np.dot(W2.T, g)
    h[h > 0] = 1  # ReLU
    # h = sigmoid(Compute_S(W1,b1,input_data))*sigmoid(1-Compute_S(W1, b1, input_data))   #Sigmoid
    g = g * h
    grad_b1 = np.mean(g,1)
    grad_W1 = np.dot(g,input_data.T)/batch_size + 2 * lam * W1

    grad_b1 = np.reshape(grad_b1,[m,1])
    grad_b2 = np.reshape(grad_b2,[10,1])
    return grad_W1, grad_W2, grad_b1, grad_b2


def ComputeAccuracy(P, input_label_no_onehot, batch_size):
    Q = P.argmax(axis=0)  # Predict label
    #print(Q)
    Y = input_label_no_onehot
    diff = Q - Y
    acc = np.sum(diff == 0)/batch_size
    return acc

def Compute_momentum(grad_W1, grad_W2, grad_b1, grad_b2, v_b1, v_b2, v_W1, v_W2):
    v_b1 = rho * v_b1 + learning_rate * grad_b1
    v_b2 = rho * v_b2 + learning_rate * grad_b2
    v_W1 = rho * v_W1 + learning_rate * grad_W1
    v_W2 = rho * v_W2 + learning_rate * grad_W2
    return v_b1, v_b2, v_W1, v_W2

def ComputeGradients_correct(W1, W2, b1, b2, P, input_data, input_label, lam, batch_size, m, h):
    small = 0.00000001
    s1 = Compute_S(W1,b1,input_data)
    h = Compute_h(s1)
    s2 = Compute_S(W2, b2, h)
    P = EvaluationClassifier(s2)
    [c,_] = ComputeCost(input_label, lam, batch_size, P,W1,W2)
    grad_b1 = np.zeros(np.shape(b1))
    grad_b2 = np.zeros(np.shape(b2))
    grad_W1 = np.zeros(np.shape(W1))
    grad_W2 = np.zeros(np.shape(W2))
    print(b1[0])
    for i in range(np.size(b1)):
        b1_try = b1

        b1_try[i] = b1_try[i] + small

        s1 = Compute_S(W1,b1_try,input_data)
        h = Compute_h(s1)
        s2 = Compute_S(W2, b2, h)
        P = EvaluationClassifier(s2)
        [c2,_] = ComputeCost(input_label, lam, batch_size, P,W1,W2)
        b1[i] = b1[i] - small
        grad_b1[i] = (c2-c)/small


    for i in range(np.size(b2)):
        b2_try = b2
        b2_try[i] = b2_try[i] + small

        s1 = Compute_S(W1,b1,input_data)
        h = Compute_h(s1)
        s2 = Compute_S(W2, b2_try, h)
        P = EvaluationClassifier(s2)
        [c2,_] = ComputeCost(input_label, lam, batch_size, P,W1,W2)
        b2[i] = b2[i] - small
        grad_b2[i] = (c2-c)/small

    for i in range(np.shape(W1)[0]):
        for j in range(np.shape(W1)[1]):

            W_try = W1
            W_try[i][j] = W_try[i][j] + small
            s1 = Compute_S(W_try,b1,input_data)
            h = Compute_h(s1)
            s2 = Compute_S(W2, b2, h)
            P = EvaluationClassifier(s2)
            [c2,_] = ComputeCost(input_label, lam, batch_size, P,W_try,W2)
            grad_W1[i][j] = (c2-c)/small
            W1 = W1 - small

    for i in range(np.shape(W2)[0]):
        for j in range(np.shape(W2)[1]):
            W_try = W2
            W_try[i][j] = W_try[i][j] + small
            s1 = Compute_S(W1,b1,input_data)
            h = Compute_h(s1)
            s2 = Compute_S(W_try, b2, h)
            P = EvaluationClassifier(s2)
            [c2,_] = ComputeCost(input_label, lam, batch_size, P,W1,W_try)
            grad_W2[i][j] = (c2-c)/small
            W2 = W2 - small
    return grad_W1, grad_W2, grad_b1, grad_b2

#Parameter
batch_size = 100
lam = 0.004
learning_rate = 0.028
rho = 0.9
m = 10 #number of hidden nodes
MAX = 20
decay_rate = 0.95
training_data = 10000
W_b_dimension = np.array([100,100,100])
#Load data and initialization

[data_1, label_1, data_length_1, label_no_onehot_1] = LoadBatch("data_batch_1.mat")
[data_2, label_2, data_length_2, label_no_onehot_2] = LoadBatch("test_batch.mat")

data_1_mean = np.mean(data_1,1)
data_1_mean = np.reshape(data_1_mean,[3072,1])
data_1 = data_1 - data_1_mean
data_2 = data_2 - data_1_mean


#Start training!

value = []

lr_max = 0.05
lr_min = 0.025

lam_max = 0.002
lam_min = 0.0

for j in range(1):

#    learning_rate = np.random.uniform(lr_min, lr_max)
#    lam = np.random.uniform(lam_min, lam_max)
    [list_W, list_b] = initialization(W_b_dimension)
    exit()
    lr = learning_rate # Store the origin learning rate before weight decay.
    J_store_1 = []
    J_store_2 = []
    loss_store_1 = []
    loss_store_2 = []
    acc_1 = []
    acc_2 = []
    v_b1 = 0
    v_b2 = 0
    v_W1 = 0
    v_W2 = 0 #Initialization of momentum

    for epoch in range(MAX):
        learning_rate = learning_rate * decay_rate
        #print(epoch)
            #print("This is epoch",epoch)
            #learning_rate = learning_rate * 0.9
        for i in range(int(training_data/batch_size)):
        #for i in range(1):

            input_data = data_1[:,i*batch_size:(i+1)*batch_size]
            input_label = label_1[:,i*batch_size:(i+1)*batch_size]
            input_label_no_onehot = label_no_onehot_1[i*batch_size:(i+1)*batch_size]
            s1 = Compute_S(W1, b1, input_data)
            h = Compute_h(s1) #RELU
        #    h = sigmoid(s1)
            s2 = Compute_S(W2,b2,h)
            P = EvaluationClassifier(s2)  # 10 * Batch_size
            grad_W1, grad_W2,  grad_b1, grad_b2 = ComputeGradients(W1, W2, b1, b2, P, input_data, input_label, lam, batch_size, m, h)
            grad_W1_x, grad_W2_x,  grad_b1_x, grad_b2_x = ComputeGradients_correct(W1, W2, b1, b2, P, input_data, input_label, lam, batch_size, m, h)

            print((grad_b1[0]-grad_b1_x[0])/grad_b1[0])
            [v_b1, v_b2, v_W1, v_W2] = Compute_momentum(grad_W1, grad_W2, grad_b1, grad_b2, v_b1, v_b2, v_W1, v_W2)
            W1 = W1 - v_W1
            W2 = W2 - v_W2
            b1 = b1 - v_b1
            b2 = b2 - v_b2


    #    s1 = Compute_S(W1,b1,data_1)
        s1 = Compute_S(W1,b1,input_data)
        h = Compute_h(s1) #RELU
    #    h = sigmoid(s1) #sigmoid
        s2 = Compute_S(W2, b2, h)
        P_use = EvaluationClassifier(s2)
    #    J,loss = ComputeCost(label_1, lam, data_length_1, P_use,W1,W2)
    #    acc = ComputeAccuracy(P_use,label_no_onehot_1, data_length_1)
        J,loss = ComputeCost(input_label, lam, 100 , P_use,W1,W2)
        acc = ComputeAccuracy(P_use,input_label_no_onehot, 100)
        J_store_1.append(J)
        acc_1.append(acc)
        loss_store_1.append(loss)

            # We run our model on validation set


        s1 = Compute_S(W1,b1,data_2)
        h = Compute_h(s1) #reLu
    #    h = sigmoid(s1) #Sigmoid
        s2 = Compute_S(W2, b2, h)
        P_use = EvaluationClassifier(s2)

        J,loss = ComputeCost(label_2, lam, data_length_2, P_use,W1,W2)
        acc = ComputeAccuracy(P_use, label_no_onehot_2, data_length_2)
        J_store_2.append(J)
        acc_2.append(acc)
        loss_store_2.append(loss)
        #print(epoch, acc)



x_axis = range(MAX)

plt.figure(1)
plt.xlabel("epoch")
plt.ylabel("cost")
plt.plot(x_axis,J_store_1,'r',label='training data')
plt.plot(x_axis,J_store_2,'g',label='validation data')

plt.legend()
plt.savefig('cost.png')

plt.figure(2)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.plot(x_axis,acc_1,'r',label='training data')
plt.plot(x_axis,acc_2,'g',label='validation data')
plt.legend()
plt.savefig('accuracy.jpg')

plt.figure(3)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(x_axis,loss_store_1,'r',label='training data')
plt.plot(x_axis,loss_store_2,'g',label='validation data')

plt.legend()
plt.savefig('loss.png')


plt.show()
