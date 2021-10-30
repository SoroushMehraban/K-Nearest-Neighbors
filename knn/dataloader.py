import pandas as pd


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
    dataframe = pd.read_csv('knn/iris.csv', names=['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class'])
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
    dataframe = pd.read_csv('knn/heart.csv')
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
    dataframe = pd.read_csv('knn/campus_recruitment.csv')

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
    dataframe = pd.read_csv('knn/healthcare-dataset-stroke-data.csv')
    dataframe.drop(['id'], axis=1, inplace=True)  # remove id since it's not useful

    """
    Encode the objects as an enumerated type
    """
    for column_name in dataframe:
        column_is_not_numeric = dataframe[column_name].dtypes != 'float64'
        if column_is_not_numeric:
            dataframe[column_name] = pd.factorize(dataframe[column_name])[0]

    return dataframe
