import numpy as np

class FeedforwardNeuralNetwork():
    def __init__(self,max_iter=100,min_konv=1, lernrate=0.8):
        NEURONS=[3]
        self.HL = len(NEURONS) #Hidden Layers
        self.NEURONS = NEURONS #Neurons in Hidden Layers

        self.max_iter = max_iter
        self.min_konv = min_konv
        self.lernrate=lernrate
    
    def ReLUFkt(self,X):
        #ReLU-Funktion is a Activation function
        #Used to transforme the data in the funcion
        return np.maximum(0,X)

    def fit(self, X_train, y_train):
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()

        #Initialise weights and bias with random numbers
        #Input Layer
        self.a0=X_train.T #Collumn needs to be the Features of datapoint
        #HL1
        self.W1T=np.random.rand(len(self.a0),self.NEURONS[0]).T
        self.b1=np.random.rand(self.NEURONS[0])
        #Output
        self.W2T=np.random.rand(self.NEURONS[0],1).T
        self.b2=np.random.rand(1,)

        for _ in range(self.max_iter):
            #Get Prediction
            #HL1
            self.z1=np.dot(self.W1T,self.a0)+self.b1[:, np.newaxis]
            self.a1=self.ReLUFkt(self.z1)
            #Output
            self.z2=np.dot(self.W2T,self.a1)+self.b2#Identification Funktion for Regression
            prediction = self.z2

            #length train data
            n = len(y_train)
            #Mathematical Derivatives
            dZ2 = prediction-y_train.T
            dW2 = (1/n)*np.dot(dZ2,self.a1.T)
            db2 = np.mean(dZ2)
            dZ1 = np.dot(self.W2T.T,dZ2)*np.where(self.z1 > 0, 1, 0)
            dW1 = (1/n)*np.dot(dZ1,self.a0.T)
            db1 = np.mean(dZ1)

            #Update weights and bias
            self.W2T=self.W2T-self.lernrate*dW2
            self.b2=self.b2-self.lernrate*db2

            self.W1T=self.W1T-self.lernrate*dW1
            self.b1=self.b1-self.lernrate*db1

    def predict(self,X_test):
        #HL1
        self.z1=np.dot(self.W1T,X_test.to_numpy().T)+self.b1[:, np.newaxis]
        self.a1=self.ReLUFkt(self.z1)

        #Output
        self.z2=np.dot(self.W2T,self.a1)+self.b2
        output=self.z2 #Identification Funktion for Regression
        #return output
        return np.clip(np.round(output),0,10)