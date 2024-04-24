import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class HierarchicalModel:
    def __init__(self, df):
        df.rename(columns={'y2': 'Type 2', 'y3': 'Type 3','y4':'Type 4'}, inplace=True)
        if 'Type 2' not in df.columns:
            raise KeyError("'Type 2' column not found in the dataset.")
        
        self.df = self.preprocess_data(df)
        self.classes_type_2 = df['Type 2'].unique()
        self.random_forest_models_type_2 = {}
        self.train_models()

    def preprocess_data(self, df):

        # Rename columns
        #df.rename(columns={'y2': 'Type 2', 'y3': 'Type 3','y4':'Type 4'}, inplace=True)
        label_encoder = LabelEncoder()
        df['Type 2'] = label_encoder.fit_transform(df['Type 2'])
        df['Type 3'] = label_encoder.fit_transform(df['Type 3'])
        df['Type 4'] = label_encoder.fit_transform(df['Type 4'])
        return df
    

    def train_models(self):
        for class_type_2 in self.classes_type_2:
            self.random_forest_models_type_2[class_type_2] = {}
            X_filtered_type_2 = self.df[self.df['Type 2'] == class_type_2][['Type 2']].values
            if len(X_filtered_type_2) > 0:  # Check if there are samples available
                for class_type_3 in self.df['Type 3'].unique():
                    X_filtered_type_3 = self.df[(self.df['Type 2'] == class_type_2) & (self.df['Type 3'] == class_type_3)][['Type 3']].values
                    y_filtered_type_4 = self.df[(self.df['Type 2'] == class_type_2) & (self.df['Type 3'] == class_type_3)]['Type 4'].values
                    if len(X_filtered_type_3) > 0:  # Check if there are samples available
                        rf_model_type_3 = RandomForestClassifier()
                        rf_model_type_3.fit(X_filtered_type_3, y_filtered_type_4)
                        self.random_forest_models_type_2[class_type_2][class_type_3] = rf_model_type_3
                    else:
                        print(f"No samples available for Type 3: {class_type_3} under Type 2: {class_type_2}")
            else:
                print(f"No samples available for Type 2: {class_type_2}")

    def predict(self, new_instance_type_2):
        # Convert the dictionary to a tuple of its items
        new_instance_key = tuple(new_instance_type_2.items())
        new_instance_features = np.array([new_instance_type_2]).reshape(1, -1)
        # Check if the tuple key exists in self.random_forest_models_type_2
        if new_instance_key in self.random_forest_models_type_2:
            rf_models_type_3 = self.random_forest_models_type_2[new_instance_key]
            predictions = {}
            for class_type_3, rf_model_type_3 in rf_models_type_3.items():
                # Make predictions for each class_type_3
                # Add your prediction logic here
                predictions[class_type_3] = rf_model_type_3.predict(new_instance_features)
            return predictions
        else:
            print("No RandomForest models found for the given class in Type 2.")
            return None




