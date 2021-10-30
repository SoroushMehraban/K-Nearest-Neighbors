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


def get_iris_dataset():
    """
    Columns:
    - Sepal length in cm.
    - Sepal width in cm.
    - Petal length in cm.
    - Petal width in cm.
    - Class: Setosa, Versicolor, and virginica

    In this practice, class is predicted using other features
    """
    dataframe = pd.read_csv('iris.csv', names=['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class'])
    dataframe['Class'] = pd.factorize(dataframe['Class'])[0]
    return dataframe


def get_heart_disease_uci_dataset():
    """
    dataset is downloaded from https://www.kaggle.com/ronitf/heart-disease-uci

    Columns:
    - age: age in years
    - sex: 1 = male, 0 = female
    - cp: chest pain type
    - trestbps: resting blood pressure (in mm Hg on admission to the hospital)
    - chol: serum cholestoral in mg/dl
    - fbs: fasting blood sugar &gt; 120 mg/d. (1= true, 0 = false)
    - restecg: resting electrocardiographic results
    - thalach: maximum heart rate achieved
    - exang: exercise induced angina (1 = yes; 0 = no)
    - oldpeak: ST depression induced by exercise relative to rest
    - slope: the slope of the peak exercise ST segment
    - ca: number of major vessels (0-3) colored by flourosopy
    - thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
    - target: 1 = has heart disease, 0 = doesn't have heart disease

    In this practice, target is predicted using other features
    """
    dataframe = pd.read_csv('heart.csv')
    return dataframe


def get_campus_recruitment_dataset():
    """
    dataset from https://www.kaggle.com/benroshan/factors-affecting-campus-placement

    Columns:
    - gender
    - ssc_p: Secondary Education percentage- 10th Grade
    - ssc_b: Board of Education- Central/ Others
    - hsc_p: Higher Secondary Education percentage- 12th Grade
    - hsc_b: Board of Education- Central/ Others
    - hsc_s: Specialization in Higher Secondary Education
    - degree_p: Degree Percentage
    - degree_t: Under Graduation(Degree type)- Field of degree education
    - etest_p: Employability test percentage ( conducted by college)
    - specialisation: Post Graduation(MBA)- Specialization
    - mba_p: MBA percentage
    - status: Status of placement- Placed/Not placed
    - workex: Work Experience

    In this practice, work experience is predicted using other features
    """
    dataframe = pd.read_csv('campus_recruitment.csv')

    dataframe.drop(['sl_no'], axis=1, inplace=True)  # remove serial number since it's not useful

    """
    Encode the objects as an enumerated type
    """
    for column_name in dataframe:
        column_is_not_numeric = dataframe[column_name].dtypes != 'float64'
        if column_is_not_numeric:
            dataframe[column_name] = pd.factorize(dataframe[column_name])[0]

    dataframe.dropna(inplace=True)  # Remove missing values

    """
    Move Work experience to the last column since we want to predict that
    """
    work_experience_column = dataframe['workex']
    dataframe.drop(['workex'], axis=1, inplace=True)  # remove serial number since it's not useful
    dataframe['workex'] = work_experience_column
    return dataframe


def get_stroke_prediction_dataset():
    """
    dataset from https://www.kaggle.com/fedesoriano/stroke-prediction-dataset

    Columns:
    - gender
    - age
    - hypertension: Hypertension binary feature
    - heart_disease: Heart disease binary feature
    - ever_married: Has the patient ever been married?
    - work_type: Work type of the patient
    - residence_type: Residence type of the patient
    - avg_glucose_level: Average glucose level in blood
    - bmi: Body Mass Index
    - smoking_status: Smoking status of the patient
    - stroke: Stroke event

    In this practice, stroke event is predicted using other features
    """
    dataframe = pd.read_csv('healthcare-dataset-stroke-data.csv')
    dataframe.drop(['id'], axis=1, inplace=True)  # remove id since it's not useful

    """
    Encode the objects as an enumerated type
    """
    for column_name in dataframe:
        column_is_not_numeric = dataframe[column_name].dtypes != 'float64'
        if column_is_not_numeric:
            dataframe[column_name] = pd.factorize(dataframe[column_name])[0]

    return dataframe


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
