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
#    list_b.append(np.random.normal(0,0.00001,[W_b_dimension[0],1]))
    list_b.append(np.zeros([W_b_dimension[0],1]))
    for i in range(1,np.size(W_b_dimension)):
        list_W.append(np.random.normal(0,0.001,[W_b_dimension[i-1],W_b_dimension[i]]))
#        list_b.append(np.random.normal(0,0.00001,[W_b_dimension[i],1]))
        list_b.append(np.zeros([W_b_dimension[i],1]))
    return list_W, list_b

def ComputeCost(label, lam, batch_size, P, list_W):
    Y = label
    loss = -(1.0 / batch_size) * np.sum(Y * np.log(P))
    J = loss
    for i in range(len(list_W)):
        J = J + lam * (np.sum(np.power(list_W[i],2)))
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

def Compute_P(list_W, list_b, input_data):
    list_s = []
    list_x = []
    list_x.append(input_data)
    Num = len(list_W)
    for i in range(Num-1):
        s = Compute_S(list_W[i].T, list_b[i], list_x[i])
        mu = np.mean(s,1)
        mu = np.reshape(mu,[np.size(mu),1])
        v = np.mean(np.power(s-mu,2),1)
        v = np.reshape(v,[np.size(v),1])
#        s = BatchNormalize(s,mu,v)
        h = Compute_h(s) #RELU    # h = sigmoid(s)
        list_x.append(h)
    s = Compute_S(list_W[Num-1].T, list_b[Num-1], h)
    list_s.append(s)
    P = EvaluationClassifier(s)  # 10 * Batch_size
    return P
def BatchNormalize(s,mu,v):
    s_hat = (s-mu)/np.sqrt(np.diag(v+0.0001)) #In case of dividing by 0
    return s_hat

def ComputeGradients(list_W, list_b, input_data, input_label, lam, batch_size):

    # input_data with 3072 * Batch_size
    # input_label with 10 * Batch_size
    list_s = []
    list_x = []
    list_x.append(input_data)
    Num = len(list_W)
    for i in range(Num-1):
        s = Compute_S(list_W[i].T, list_b[i], list_x[i])
        mu = np.mean(s,1)
        mu = np.reshape(mu,[np.size(mu),1])
        v = np.mean(np.power(s-mu,2),1)
        v = np.reshape(v,[np.size(v),1])
#        s = BatchNormalize(s,mu,v)
        h = Compute_h(s) #RELU    # h = sigmoid(s)
        list_x.append(h)
    s = Compute_S(list_W[Num-1].T, list_b[Num-1], h)
    list_s.append(s)
    P = EvaluationClassifier(s)  # 10 * Batch_size

    g = P - input_label
    grad_b_list = []
    grad_W_list = []
    for i in range(Num)[::-1]:
        W = list_W[i]
        h = list_x[i]
        grad_b = np.mean(g,1)
        grad_b = np.reshape(grad_b,[np.size(grad_b),1])
        grad_W = np.transpose(np.dot(g,h.T)/batch_size + 2 * lam * W.T)
        grad_b_list.append(grad_b)
        grad_W_list.append(grad_W)
        g = np.dot(W, g)
        h[h > 0] = 1  # ReLU
        # h = sigmoid(Compute_S(W1,b1,input_data))*sigmoid(1-Compute_S(W1, b1, input_data))   #Sigmoid
        g = g * h
    grad_b_list = grad_b_list[::-1]
    grad_W_list = grad_W_list[::-1]
    return grad_W_list, grad_b_list


def ComputeAccuracy(P, input_label_no_onehot, batch_size):
    Q = P.argmax(axis=0)  # Predict label
#print(Q)
    Y = input_label_no_onehot
    diff = Q - Y
    acc = np.sum(diff == 0)/batch_size
    return acc

def Compute_momentum(grad_W_list, grad_b_list, v_W_list, v_b_list):
    for i in range(len(grad_b_list)):
        v_b_list[i] = rho * v_b_list[i] + learning_rate * grad_b_list[i]

        v_W_list[i] = rho * v_W_list[i] + learning_rate * grad_W_list[i]
    return v_W_list, v_b_list

def ComputeGradients_correct(list_W, list_b, input_data, input_label, lam, batch_size, layers):
    small = 0.000001
    P = Compute_P(list_W, list_b, input_data)
    c, _ = ComputeCost(input_label, lam, batch_size, P, list_W)
    grad_W_pseudo = []
    grad_b_pseudo = []

    for i in range(layers):
        grad_W_pseudo.append(np.ones(np.shape(list_W[i])))
        grad_b_pseudo.append(np.ones(np.shape(list_b[i])))

    for j in range(layers):
        for i in range(np.size(list_b[j])):
            b_try = list_b
    #        print(list_b[0])
#           print(b_try[0])
            (b_try[j])[i] = (b_try[j])[i] + small
            P = Compute_P(list_W, b_try, input_data)
   #          print(P)

	#       print(b_try[0])
            c2, _ = ComputeCost(input_label, lam, batch_size, P, list_W)

        #    print(list_b[0])
            (grad_b_pseudo[j])[i] = (c2-c)/small
            (b_try[j])[i] = (b_try[j])[i] - small

    return grad_W_pseudo, grad_b_pseudo

'''
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
            '''


#Parameter
batch_size = 1
lam = 0.0
learning_rate = 0.05
rho = 0.9
MAX = 20
decay_rate = 0.9
training_data = 10000
W_b_dimension = np.array([50,10])
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
    lr = learning_rate # Store the origin learning rate before weight decay.
    J_store_1 = []
    J_store_2 = []
    loss_store_1 = []
    loss_store_2 = []
    acc_1 = []
    acc_2 = []
    v_b_list = (np.zeros(np.size(W_b_dimension))).tolist()
    v_W_list = (np.zeros(np.size(W_b_dimension))).tolist() #Initialization of momentum

    for epoch in range(MAX):
        learning_rate = learning_rate * decay_rate
        print("This is epoch",epoch)
        learning_rate = learning_rate * 0.9
        for i in range(int(training_data/batch_size)):
        #for i in range(1):
            layers = 2
            input_data = data_1[:,i*batch_size:(i+1)*batch_size]
            input_label = label_1[:,i*batch_size:(i+1)*batch_size]
            input_label_no_onehot = label_no_onehot_1[i*batch_size:(i+1)*batch_size]
            grad_W_list, grad_b_list = ComputeGradients(list_W, list_b, input_data, input_label, lam, batch_size)

            grad_W_pseudo, grad_b_pseudo = ComputeGradients_correct(list_W, list_b, input_data, input_label, lam, batch_size, layers)
            print((grad_b_list[1]-grad_b_pseudo[1])/grad_b_list[1])
    
            exit()
            [v_W_list, v_b_list] = Compute_momentum(grad_W_list, grad_b_list, v_W_list, v_b_list)
            for i in range(len(v_b_list)):
                list_W[i] = list_W[i] - v_W_list[i]
                list_b[i] = list_b[i] - v_b_list[i]

        P = Compute_P(list_W, list_b, data_1)
        J,loss = ComputeCost(label_1, lam, data_length_1, P, list_W)
        acc = ComputeAccuracy(P,label_no_onehot_1, data_length_1)
        J_store_1.append(J)
        acc_1.append(acc)
        loss_store_1.append(loss)
        print("Training acc: ",acc)
            # We run our model on validation set

        P = Compute_P(list_W, list_b, data_2)
        J,loss = ComputeCost(label_2, lam, data_length_2, P, list_W)
        acc = ComputeAccuracy(P,label_no_onehot_2, data_length_2)
        J_store_2.append(J)
        acc_2.append(acc)
        loss_store_2.append(loss)
        print("Validation acc: ",acc)



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
