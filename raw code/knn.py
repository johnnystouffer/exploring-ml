import numpy as np
import matplotlib.pyplot as plt

class KNN:
    
    def __init__(self, k=3):
        if self.k % 2:
            raise ValueError("k must be an odd number")
        else:
            self.k = k
            self.x_test = None
            
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.y_predict = []
        
    def euclidean_distance(self, points1, points2):
        return np.sqrt(np.sum((points1 - points2)**2, axis=1))
    
    def predict(self, x_test):
        self.x_test = x_test
        for test_point in x_test:
            distances = []
            for train_point, label in zip(self.x_train, self.y_train):
                distance = self.euclidean_distance(test_point, train_point)
                distances.append((distance, label))
            neighbors = sorted(distances)[:self.k]
            labels = [label for _, label in neighbors]
            counts = {label: labels.count(label) for label in set(labels)}
            best_neighbor = max(counts, key=counts.get)
            self.y_predict.append(best_neighbor)
        return self.y_predict
    
    def evaluation(self, y_test):
        true_positive = sum((y_test == 1) & (self.y_predict == 1))
        false_positive = sum((y_test == 0) & (self.y_predict == 1))
        true_negative = sum((y_test == 0) & (self.y_predict == 0))
        false_negative = sum((y_test == 1) & (self.y_predict == 0))
        
        accuracy = (true_positive + true_negative) / len(y_test)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        
        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)