import numpy as np

class NaiveBayesKlassifikator():
    def __init__(self):
        self.P_C = {}
        self.count_features_in_label = {}

    def fit(self, X_train, y_train):
        self.labels = set(y_train)
        self.X_train = X_train
        self.y_train = y_train

        features = [0 for _ in X_train[0]]
        
        for label in self.labels:
            p_c = y_train.count(label)/len(y_train)
            self.P_C[label]=p_c

            self.count_features_in_label[label]=features

        for i, datapoint in enumerate(X_train):
            self.count_features_in_label[y_train[i]] = np.array(self.count_features_in_label[y_train[i]]) + np.array(datapoint)

    def predict(self, X_pred):
        pred =[]
        
        #go threw every datapoint
        for datapoint in X_pred:
            P_C_X_list = []
            for label in self.labels:
                #calculate log(P(X|Ck))
                p_X_C = 0
                for i, xi in enumerate(datapoint):
                    if xi != 0:
                        #calculate log(P(xi|Ck))
                        p_xi_bed_c = (self.count_features_in_label[label][i]+1)/(self.count_features_in_label[label].sum()+len(datapoint))
                        p_X_C+=np.log(p_xi_bed_c)**xi
                
                #P(C|X)∝(log(P(Ck))+log(P(X|Ck))
                p_C_X = np.log(self.P_C[label])+p_X_C

                P_C_X_list.append(p_C_X)
            
            #argmaxCk(log(P(Ck))+log(P(X|Ck))
            pred.append(list(self.labels)[P_C_X_list.index(max(P_C_X_list))])
        return(pred)
