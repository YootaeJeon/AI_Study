import numpy as np 

class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out   

class TwoLayerNet:
    def __init__(self, input, hidden, output):
        I, H, O = input,hidden, output

        W1 = np.random.randn(I,H)  
        b1 = np.random.randn(H)
        W2 = np.random.randn(H,O)
        b2 = np.random.randn(O)

        #계층 생성
        self.layers = [
            Affine(W1,b1),  # return x (10,2)   w1(2,4) b1(4)  > h (10,4)
            Sigmoid(),
            Affine(W2,b2)  # return h (10,4)    w2(4,3) b2(3)
        ]

        #모든 가중치를 리스트에 모은다.
        self.params = []                  # a = ['a','b]     
        for layer in self.layers:         # a+=['c','d'] >>> ['a','b','c','d']
            self.params += layer.params   #  # Affine> Sigmoid > Affine
  
    def predict(self,x):
        for layer in self.layers:  # 각 계층의 forward의 out값이 각각 나옴 
            print('계층 함수의 forward :\n',x.shape)
            x = layer.forward(x)    #Input (10,2) >>> 
                                    #Affine.forward 뉴런 4> out (10,4) >>> 
                                    #Sigmoid.forward >>>  (10,4)
                                    #Affine.forward > out (4.3)
        return x
        

x = np.random.randn(10,2)
model = TwoLayerNet(2,4,3)
s = model.predict(x)
print('최종',s)