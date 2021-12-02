import numpy as np
import pandas as pd
import timeit


class KNearestNeighbor:
    def __init__(self, dataset, k):
        self.train_data, self.test_data = self.train_test_split(dataset)
        self.k = k

    def euclidean_distance(self, row1, row2):
        return np.sqrt(np.sum(np.power(row1 - row2, 2)))

    def train_test_split(self, dataframe):
        np.random.seed(0)
        mask = np.random.rand(len(dataframe)) < 0.8
        return dataframe[mask].to_numpy(), dataframe[~mask].to_numpy()

    def get_k_neighbors(self, target_row):
        distances = []
        for train_row in self.train_data:
            test_dataset_has_label = train_row.size == target_row.size
            if test_dataset_has_label:
                distance = self.euclidean_distance(train_row[:-1], target_row[:-1])  # every column except label
            else:
                distance = self.euclidean_distance(train_row[:-1], target_row)
            distances.append(distance)

        sorted_indexes = sorted(range(len(self.train_data)), key=lambda i: distances[i])
        neighbors = self.train_data[sorted_indexes[:self.k]]
        return neighbors

    def predict_class(self, target_row):
        neighbors = self.get_k_neighbors(target_row)
        output_values = neighbors[:, -1]

        unique_output_values, counts = np.unique(output_values, return_counts=True)
        most_frequent_index = np.argmax(counts)
        most_frequent_output_value = unique_output_values[most_frequent_index]
        return most_frequent_output_value

    def loss(self, predictions):
        test_labels = self.test_data[:, -1]
        true_estimations = np.count_nonzero(predictions == test_labels)
        return (true_estimations / len(self.test_data)) * 100

    def predict(self):
        predictions = []
        for test_row in self.test_data:
            prediction = self.predict_class(test_row)
            predictions.append(prediction)
        predictions = np.array(predictions)
        accuracy = self.loss(predictions)
        return accuracy
