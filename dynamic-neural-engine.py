import numpy as np
import matplotlib.pyplot as plt
class classifier:
    def __init__(self,inp,hid,out,layers):
        self.w_lst = []
        self.b_lst = []
        self.h_lst = []
        self.z_lst = []
        self.de_lst = []
        self.v_lst = []
        self.bv_lst = []
        self.wf_lst = []
        self.bf_lst = []
        self.loss = []
        
        for i in range(0,layers+1):
            if i ==0:
                w = np.random.randn(inp,hid)*np.sqrt(2/inp)
                self.w_lst.append(w)
                b = np.zeros((1,hid))
                self.b_lst.append(b)
                v = np.zeros_like(w)
                self.v_lst.append(v)
                bv = np.zeros_like(b)
                self.bv_lst.append(bv)
                wf = np.zeros_like(w)
                self.wf_lst.append(wf)
                bf = np.zeros_like(b)
                self.bf_lst.append(bf)
            elif i == layers:
                w = np.random.randn(hid,out)*np.sqrt(2/hid)
                self.w_lst.append(w)
                b = np.zeros((1,out))
                self.b_lst.append(b)
                v = np.zeros_like(w)
                self.v_lst.append(v)
                bv = np.zeros_like(b)
                self.bv_lst.append(bv)
                wf = np.zeros_like(w)
                self.wf_lst.append(wf)
                bf = np.zeros_like(b)
                self.bf_lst.append(bf)
            else:
                w = np.random.randn(hid,hid)*np.sqrt(2/hid)
                self.w_lst.append(w)
                b = np.zeros((1,hid))
                self.b_lst.append(b)
                v = np.zeros_like(w)
                self.v_lst.append(v)
                bv = np.zeros_like(b)
                self.bv_lst.append(bv)
                wf = np.zeros_like(w)
                self.wf_lst.append(wf)
                bf = np.zeros_like(b)
                self.bf_lst.append(bf)
    def relu(self,x):
        return np.where(x>0,x,0.01*x)
    def relu_derivative(self,x):
        return np.where(x>0,1,0.01)
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def train(self,x,y,lr = 0.01,g = 0.9,epochs = 10000):
        for i in range(epochs + 1):
           for n in range(len(self.w_lst)):
               self.wf_lst[n] = self.w_lst[n] - self.v_lst[n]*g
               self.bf_lst[n] = self.b_lst[n] - self.bv_lst[n]*g
           h = x@self.w_lst[0] + self.b_lst[0]
           self.h_lst.append(h)
           z = self.relu(h + x)
           self.z_lst.append(z)
           for j in range(1,len(self.w_lst)):
                h = z@self.wf_lst[j] + self.bf_lst[j]
                self.h_lst.append(h)
                if j == len(self.w_lst) - 1:
                   y_pred = self.sigmoid(h)
                else:
                   z = self.relu(h+x)
                   self.z_lst.append(z)
           doe = y_pred - y
           self.de_lst.append(doe)
           for k in range(1,len(self.w_lst)):
                he = doe@self.wf_lst[-k].T
                doe = he*self.relu_derivative(self.h_lst[-k-1] + 1)
                self.de_lst.append(doe)
           self.de_lst = self.de_lst[::-1]
           grad = x.T@self.de_lst[0]
           self.v_lst[0] = grad*lr + self.v_lst[0]*g
           self.w_lst[0]-=self.v_lst[0]
           grad = np.sum(self.de_lst[0],axis=0,keepdims=True)
           self.bv_lst[0] = grad*lr + self.bv_lst[0]*g
           self.b_lst[0]-=self.bv_lst[0]
           for l in range(1,len(self.w_lst)):
               grad = self.z_lst[l-1].T@self.de_lst[l]
               self.v_lst[l] = grad*lr + self.v_lst[l]*g
               self.w_lst[l]-=self.v_lst[l]
               grad = np.sum(self.de_lst[l],axis=0,keepdims=True)
               self.bv_lst[l] = grad*lr + self.bv_lst[l]*g
               self.b_lst[l]-=self.bv_lst[l]
           if i %1000 == 0:
               l = np.mean(np.square(y_pred - y))
               print(f"the loss is {l}")
               self.loss.append(l)
           self.h_lst = []
           self.z_lst = []
           self.de_lst = []
       
    def plot(self):
        plt.plot(self.loss)
        plt.xlabel("Iterations")
        plt.ylabel("loss (MSE)")
        plt.grid(True)
        plt.show()
x = np.array([[0,0,0],[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0],[1,1,1]])
y = np.array([[0,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[1,1]])
model = classifier(3,3,2,2)
model.train(x,y)
model.plot()



