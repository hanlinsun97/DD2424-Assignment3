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
    list_W.append(np.random.randn(3072,W_b_dimension[0])/np.sqrt(3072/2))
#    list_b.append(np.random.normal(0,0.00001,[W_b_dimension[0],1]))
    list_b.append(np.zeros([W_b_dimension[0],1]))
    for i in range(1,np.size(W_b_dimension)):
        list_W.append(np.random.randn(W_b_dimension[i-1],W_b_dimension[i])/np.sqrt(W_b_dimension[i-1]/2))
#        list_b.append(np.random.normal(0,0.00001,[W_b_dimension[i],1]))
        list_b.append(np.zeros([W_b_dimension[i],1]))
    return list_W, list_b

def ComputeCost(label, lam, batch_size, P, list_W):
    Y = label
    loss = -(1.0 / batch_size) * np.sum(Y * (np.log(P)))
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
def Compute_Ind_s(s):
    s_ind = s
    for i in range(np.shape(s_ind)[0]):
        for j in range(np.shape(s_ind)[1]):
            if s_ind[i][j] > 0 :
                s_ind[i][j] = 1
            else:
                s_ind[i][j] = 0
    return s_ind

def Compute_P(list_W, list_b, input_data, batch_number, mu_store, v_store, alpha):
    list_x = []
    list_s = []
    list_x.append(input_data)
    Num = len(list_W)
    for i in range(Num-1):
        s = Compute_S(list_W[i].T, list_b[i], list_x[i])
        list_s.append(s)
        if(BN==1):
            mu = np.mean(s,1)
            mu = np.reshape(mu,[np.size(mu),1])
            v = np.mean(np.power(s-mu,2),1)
            v = np.reshape(v,[np.size(v),1])
            V_b = v + 0.0000001
            s = BatchNormalize(s,mu,V_b)
            if(batch_number==0):
                mu_store.append(mu)
                v_store.append(V_b)
            else:
                mu_store[i] = alpha * mu_store[i] + (1 - alpha) * mu
                v_store[i] = alpha * v_store[i] + (1 - alpha) * V_b

        h = Compute_h(s) #RELU    # h = sigmoid(s)
        list_x.append(h)

    s = Compute_S(list_W[Num-1].T, list_b[Num-1], h)
    list_s.append(s)
    P = EvaluationClassifier(s)  # 10 * Batch_size
    return P, list_x, list_s, mu_store, v_store

def Compute_P_normal(list_W, list_b, input_data):
        list_x = []
        list_x.append(input_data)
        Num = len(list_W)
        for i in range(Num-1):
            s = Compute_S(list_W[i].T, list_b[i], list_x[i])
            list_s.append(s)
            if(BN==1):
                mu = np.mean(s,1)
                mu = np.reshape(mu,[np.size(mu),1])
                v = np.mean(np.power(s-mu,2),1)
                v = np.reshape(v,[np.size(v),1])
                V_b = v + 0.0000001
                s = BatchNormalize(s,mu,V_b)
            h = Compute_h(s) #RELU    # h = sigmoid(s)
            list_x.append(h)
        s = Compute_S(list_W[Num-1].T, list_b[Num-1], h)
        list_s.append(s)
        P = EvaluationClassifier(s)  # 10 * Batch_size
        return P



def Compute_P_on_validation(list_W, list_b, input_data, mu_store, v_store):
    list_x = []
    list_s = []
    list_x.append(input_data)
    Num = len(list_W)
    for i in range(Num-1):

        s = Compute_S(list_W[i].T, list_b[i], list_x[i])
        list_s.append(s)
        if(BN==1):
            mu = mu_store[i]
            v = v_store[i]
            s = BatchNormalize(s,mu,v)

        h = Compute_h(s) #RELU    # h = sigmoid(s)
        list_x.append(h)

    s = Compute_S(list_W[Num-1].T, list_b[Num-1], h)
    list_s.append(s)
    P = EvaluationClassifier(s)  # 10 * Batch_size
    return P



def BatchNormalize(s,mu,v):
    s_hat = (s-mu)/(np.sqrt((v+0.000001))) #In case of dividing by 0

    return s_hat





def BatchNormBackPass(g, list_s, l,batch_size):   # L MEANS LAYER IN BACKPASS HERE !!!
    s = list_s[l]
    mu = np.mean(s,1)
    mu = np.reshape(mu,[np.size(mu),1])
    v = np.mean(np.power(s-mu,2),1)
    v = np.reshape(v,[np.size(v),1])
    V_b = (v + 0.0000001)

    grad_mu = - np.reshape(np.sum(g,1),[np.size(np.sum(g,1)),1]) * np.power(V_b,-1/2)
    grad_v = (-1/2) * np.reshape(np.sum(g * (s-mu),1),[np.size(np.sum(g * (s-mu),1)),1]) *np.power(V_b, -3/2)

    #print(grad_v.shape)
#    print(np.shape(np.power(V_b,-1/2)))
    g = g * np.power(V_b,-1/2) + (2/batch_size) * grad_v * (s-mu) + grad_mu * (1/batch_size)
    return g

def ComputeGradients(list_W, list_b, input_data, input_label, lam, batch_size, list_x, list_s,P):

    # input_data with 3072 * Batch_size
    # input_label with 10 * Batch_size
    # list_s = []
    # list_x = []
    # list_x.append(input_data)
    #
    # for i in range(Num-1):
    #     s = Compute_S(list_W[i].T, list_b[i], list_x[i])
    #     list_s.append(s)
    #     if(BN==1):
    #         mu = np.mean(s,1)
    #         mu = np.reshape(mu,[np.size(mu),1])
    #         v = np.mean(np.power(s-mu,2),1)
    #         v = np.reshape(v,[np.size(v),1])
    #         s_hat = BatchNormalize(s,mu,v)
    #     else:
    #         s_hat = s
    #     h = Compute_h(s_hat) #RELU    # h = sigmoid(s)
    #     list_x.append(h)
    # s = Compute_S(list_W[Num-1].T, list_b[Num-1], h)
    # list_s.append(s)
    # P = EvaluationClassifier(s)  # 10 * Batch_size
    Num = len(list_W)
    g = P - input_label
    grad_b_list = []
    grad_W_list = []

    for i in range(Num)[::-1]:

        if (i>0):
            if((BN==1) & (i < Num-1)):
                g = BatchNormBackPass(g,list_s,i,batch_size)
            W = list_W[i]

            h = list_x[i]

            grad_b = np.mean(g,1)

            grad_b = np.reshape(grad_b,[np.size(grad_b),1])
            grad_W = np.transpose(np.dot(g,h.T)/batch_size + 2 * lam * W.T)
            grad_b_list.append(grad_b)
            grad_W_list.append(grad_W)
            g = np.dot(W, g)

            s = list_s[i-1]
            mu = np.mean(s,1)
            mu = np.reshape(mu,[np.size(mu),1])
            v = np.mean(np.power(s-mu,2),1)
            v = np.reshape(v,[np.size(v),1])
            V_b = v + 0.0000001
            s = BatchNormalize(s,mu,V_b)

            s_ind = Compute_Ind_s(s)  # ReLU
            # h = sigmoid(Compute_S(W1,b1,input_data))*sigmoid(1-Compute_S(W1, b1, input_data))   #Sigmoid
            g = g * s_ind
        else:
            if(BN==1):
                g = BatchNormBackPass(g,list_s,i,batch_size)
            W = list_W[0]
            h = input_data

            grad_b = np.mean(g,1)
            grad_b = np.reshape(grad_b,[np.size(grad_b),1])
            grad_W = np.transpose(np.dot(g,h.T)/batch_size + 2 * lam * W.T)
            grad_b_list.append(grad_b)
            grad_W_list.append(grad_W)
            #g = np.dot(W, g)
            #h[h > 0] = 1  # ReLU
            # h = sigmoid(Compute_S(W1,b1,input_data))*sigmoid(1-Compute_S(W1, b1, input_data))   #Sigmoid
            #g = g * h



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
    P = Compute_P_normal(list_W, list_b, input_data)

    c, _ = ComputeCost(input_label, lam, batch_size, P, list_W)
    grad_W_pseudo = []
    grad_b_pseudo = []

    for i in range(layers):
        grad_W_pseudo.append(np.ones(np.shape(list_W[i])))
        grad_b_pseudo.append(np.ones(np.shape(list_b[i])))

    for j in range(layers):
        for i in range(np.size(list_b[j])):
            b_try = list_b
            (b_try[j])[i] = (b_try[j])[i] + small
            P = Compute_P_normal(list_W, b_try, input_data)
            c2, _ = ComputeCost(input_label, lam, batch_size, P, list_W)
            (grad_b_pseudo[j])[i] = (c2-c)/small
            (b_try[j])[i] = (b_try[j])[i] - small

    for k in range(layers):
        for i in range(np.shape(list_W[k])[0]):

            for j in range(np.shape(list_W[k])[1]):

                W_try = list_W
                W_try[k][i][j] = W_try[k][i][j] + small
                P = Compute_P_normal(W_try, list_b, input_data)
                c2, _ = ComputeCost(input_label, lam, batch_size, P, W_try)
                grad_W_pseudo[k][i][j] = (c2-c)/small
                list_W[k][i][j] = list_W[k][i][j] - small
    return grad_W_pseudo, grad_b_pseudo



#Parameter
BN = 0
batch_size = 100
lam = 0.0045
learning_rate = 0.3
rho = 0.9
MAX = 20
decay_rate = 0.85
training_data = 10000
W_b_dimension = np.array([50,10])
alpha = 0.9
#Load data and initialization

[data_1, label_1, data_length_1, label_no_onehot_1] = LoadBatch("data_batch_1.mat")
[data_2, label_2, data_length_2, label_no_onehot_2] = LoadBatch("data_batch_2.mat")
[data_3, label_3, data_length_3, label_no_onehot_3] = LoadBatch("test_batch.mat")
data_1_mean = np.mean(data_1,1)
data_1_mean = np.reshape(data_1_mean,[3072,1])
data_1 = data_1 - data_1_mean
data_2 = data_2 - data_1_mean


#Start training!

value = []
lr_max = 0.5
lr_min = 0.3
lam_max = 0.005
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
    ##    print("This is epoch",epoch)
        mu_store = []
        v_store = []
        for i in range(int(training_data/batch_size)):
            layers = 3
            input_data = data_1[:,i*batch_size:(i+1)*batch_size]
            input_label = label_1[:,i*batch_size:(i+1)*batch_size]
            input_label_no_onehot = label_no_onehot_1[i*batch_size:(i+1)*batch_size]
    #        grad_W_list = []
    #        grad_b_list = []

            P,list_x, list_s, mu_store, v_store = Compute_P(list_W, list_b, input_data, i, mu_store, v_store,alpha)
            grad_W_list, grad_b_list = ComputeGradients(list_W, list_b, input_data, input_label, lam, batch_size, list_x, list_s,P)
            # grad_W_pseudo, grad_b_pseudo = ComputeGradients_correct(list_W, list_b, input_data, input_label, lam, batch_size, layers)
            # print((grad_W_list[0][0]-grad_W_pseudo[0][0])/grad_W_list[0][0])
            # print((grad_W_list[1][1]-grad_W_pseudo[1][1])/grad_W_list[1][1])
            # print((grad_W_list[2][2]-grad_W_pseudo[2][2])/grad_W_list[2][2])
            # #print(grad_b_list[0])
            #print(np.sum(grad_W_list[0]))

        #     s = Compute_S(list_W[0].T, list_b[0], list_x[0])
        #     mu = np.mean(s,1)
        #     mu = np.reshape(mu,[np.size(mu),1])
        #     v = np.mean(np.power(s-mu,2),1)
        #     v = np.reshape(v,[np.size(v),1])
        #     s = BatchNormalize(s,mu,v)
        #     h = Compute_h(s)
        #     s = Compute_S(list_W[1].T, list_b[1], h)
        #     P = EvaluationClassifier(s)
        #     g = P- input_label
        #     mu = np.mean(s,1)
        #     mu = np.reshape(mu,[np.size(mu),1])
        #     v = np.mean(np.power(s-mu,2),1)
        #     v = np.reshape(v,[np.size(v),1])
        #     V_b = (v + 0.000001)
        #     grad_mu = - np.reshape(np.sum(g,1),[np.size(np.sum(g,1)),1]) * np.power(V_b,-1/2)
        #     grad_v = (-1/2) * np.reshape(np.sum(g * (s-mu),1),[np.size(np.sum(g * (s-mu),1)),1]) *np.power(V_b, -3/2)
        #     #print(grad_v.shape)
        # #    print(np.shape(np.power(V_b,-1/2)))
        #     g = g * np.power(V_b,-1/2) + (2/batch_size) * grad_v * (s-mu) + grad_mu * (1/batch_size)
        #     grad_W2 = np.transpose(np.dot(g,h.T)/batch_size + 2 * lam * list_W[1].T)
        #    print(grad_W_pseudo[2][2])
        #    print(grad_W_list[2][2])
            # # print(grad_W2[1])


        #    print(np.sum(np.power(grad_W2 - grad_W_list[1],2)))
        #    grad_W_list, grad_b_list = ComputeGradients(list_W, list_b, input_data, input_label, lam, batch_size)

            # grad_W_pseudo, grad_b_pseudo = ComputeGradients_correct(list_W, list_b, input_data, input_label, lam, batch_size, layers)
            # print((grad_b_list[1]-grad_b_pseudo[1])/grad_b_list[1])
            # exit()



###### BUG FREE  ###
            [v_W_list, v_b_list] = Compute_momentum(grad_W_list, grad_b_list, v_W_list, v_b_list)
            for i in range(len(v_b_list)):
                list_W[i] = list_W[i] - v_W_list[i]
                list_b[i] = list_b[i] - v_b_list[i]

    #    P_use_1,_,_ = Compute_P(list_W, list_b, data_1)
        P_use_1 = Compute_P_on_validation(list_W, list_b, data_1[:,0:training_data], mu_store, v_store)
        J,loss = ComputeCost(label_1[:,0:training_data], lam, training_data, P_use_1, list_W)
        acc = ComputeAccuracy(P_use_1,label_no_onehot_1[0:training_data], training_data)
        J_store_1.append(J)
        acc_1.append(acc)
        loss_store_1.append(loss)
        print("Training acc: ",acc)
            # We run our model on validation set

        P_use_2 = Compute_P_on_validation(list_W, list_b, data_2[:,0:training_data], mu_store, v_store)
        J,loss = ComputeCost(label_2[:,0:training_data], lam, training_data, P_use_2, list_W)
        acc = ComputeAccuracy(P_use_2,label_no_onehot_2[0:training_data], training_data)
        J_store_2.append(J)
        acc_2.append(acc)
        loss_store_2.append(loss)
        print("Validation acc: ",acc)
    #print(lr, lam, acc_1[MAX-1], acc_2[MAX-1])



x_axis = range(MAX)
print(len(x_axis))
print(len(J_store_2))
print(len(acc_1))
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
