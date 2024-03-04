import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder



def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(data):
    # Handle missing values
    data['vote_count'].fillna(data['vote_count'].median(), inplace=True)
    data['vote_average'].fillna(data['vote_average'].median(), inplace=True)
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    data['original_language'] = label_encoder.fit_transform(data['original_language'])

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = data[['vote_count', 'vote_average']]
    data[['vote_count', 'vote_average']] = scaler.fit_transform(numerical_features)

    return data


