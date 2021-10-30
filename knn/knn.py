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
            distance = self.euclidean_distance(train_row, target_row)
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




def run_algorithm(dataset, dataset_name):
    print(f"---- Running k-nearest neighbors on {dataset_name} dataset ----")
    print(f"Number of features: {len(dataset.columns)}")
    print(f"Number of rows: {len(dataset)}")

    start_time = timeit.default_timer()
    knn = KNearestNeighbor(dataset, k=10)
    accuracy = knn.predict()
    end_time = timeit.default_timer()
    print()
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Run time: {end_time - start_time:.2f} seconds")
    print()


if __name__ == '__main__':
    iris_df = get_iris_dataset()
    heart_df = get_heart_disease_uci_dataset()
    campus_recruitment_df = get_campus_recruitment_dataset()
    stroke_df = get_stroke_prediction_dataset()

    run_algorithm(dataset=iris_df, dataset_name="Iris")
    run_algorithm(dataset=heart_df, dataset_name="Heart Disease UCI")
    run_algorithm(dataset=campus_recruitment_df, dataset_name="Campus Recruitment")
    run_algorithm(dataset=stroke_df, dataset_name="Stroke Prediction")
