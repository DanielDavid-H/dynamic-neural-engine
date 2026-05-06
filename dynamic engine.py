import numpy as np
import matplotlib.pyplot as plt
x = np.array([[0,0,0],[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0],[1,1,1]])
y = np.array([[0,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[1,0]])
i_ = 3
h_ = 5 
o_ = 2
layers = int(input("enter no.of hidden_layers:"))
g = 0.9
loss =[]

w_lst = []
wf_lst = []
b_lst = []
bf_lst = []
v_lst = []
bv_lst = []
h_lst = []
z_lst = []
de_lst= []
for i in range(0,layers+1):
    if i ==0:
      w = np.random.randn(i_,h_)*np.sqrt(2/i_)
      b = np.zeros((1,h_))
      w_lst.append(w)
      b_lst.append(b)
      v = np.zeros_like(w)
      bv = np.zeros_like(b)
      v_lst.append(v)
      bv_lst.append(bv)
      wf = np.zeros_like(w)
      bf = np.zeros_like(b)
      wf_lst.append(wf)
      bf_lst.append(bf)
    elif i == layers:
        w = np.random.randn(h_,o_)*np.sqrt(2/h_)
        b = np.zeros((1,o_))
        w_lst.append(w)
        b_lst.append(b)
        v = np.zeros_like(w)
        bv = np.zeros_like(b)
        v_lst.append(v)
        bv_lst.append(bv)
        wf = np.zeros_like(w)
        bf = np.zeros_like(b)
        wf_lst.append(wf)
        bf_lst.append(bf)
    else:
        w = np.random.randn(h_,h_)*np.sqrt(2/h_)
        b = np.zeros((1,h_))
        w_lst.append(w)
        b_lst.append(b)
        v = np.zeros_like(w)
        bv = np.zeros_like(b)
        v_lst.append(v)
        bv_lst.append(bv)
        wf = np.zeros_like(w)
        bf = np.zeros_like(b)
        wf_lst.append(wf)
        bf_lst.append(bf)
        
def relu(x):
    return np.where(x>0,x,0.01*x)
def relu_derivative(x):
    return np.where(x>0,1,0.01)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return x*(1-x)

lr = 0.01
for j in range(5000):
    h_lst = []
    z_lst = []
    de_lst= []
    for n in range(len(v_lst)):
        wf_lst[n] = w_lst[n] -v_lst[n]*g
        bf_lst[n] = b_lst[n] -bv_lst[n]*g
    h = x@wf_lst[0] + bf_lst[0]
    h_lst.append(h)
    z = relu(h)
    z_lst.append(z)
    if layers == 1:
        h = z@wf_lst[1] + bf_lst[1]
        h_lst.append(h)
        y_pred = sigmoid(h)
    else:
        for i in range(1,len(w_lst)):
            h = z@wf_lst[i] + bf_lst[i]
            h_lst.append(h)
            if i != len(w_lst) - 1:
                z = relu(h)
                z_lst.append(z)
            else:
                y_pred = sigmoid(h)
    de = (y_pred - y)
    de_lst.append(de)
    for k in range(1,len(w_lst)):
        e = de@wf_lst[-k].T
        de = e*relu_derivative(h_lst[-k-1])
        de_lst.append(de)
    de_lst = de_lst[::-1]
    grad1=(x.T@de_lst[0])*lr
    v_lst[0] = grad1 + v_lst[0]*g
    w_lst[0]-=v_lst[0]
    grad2=(np.sum(de_lst[0],axis=0,keepdims=True))*lr
    bv_lst[0] = grad2 + bv_lst[0]*g
    b_lst[0]-=bv_lst[0]
    for l in range(1,len(w_lst)):
        grad=(z_lst[l-1].T@de_lst[l])
        v_lst[l] = grad*lr + v_lst[l]*g
        w_lst[l]-=v_lst[l]
        grad=(np.sum(de_lst[l],axis=0,keepdims=True))*lr
        bv_lst[l] = grad*lr + bv_lst[l]*g
        b_lst[l]-=bv_lst[l]
    if j%1000 ==0:
        l = np.mean(np.square(y_pred - y))
        loss.append(l)
        print(f"The loss is {l}")
plt.plot(loss)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.grid(True)
plt.show()

