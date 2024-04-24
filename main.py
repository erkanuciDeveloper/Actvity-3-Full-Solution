from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
from modelling.hierarchical_model import HierarchicalModel

seed = 0
random.seed(seed)
np.random.seed(seed)

def load_data():
    # Load the input data
    df = get_input_data()
    return df

def load_data_for_hierarchical():
    # Load the input data
    df = get_input_data_hierarchical()
    return df

def preprocess_data(df):
    # De-duplicate input data
    df = de_duplication(df)
   
    # Remove noise in input data
    df = noise_remover(df)
    # Translate data to English
    # df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df

def preprocess_data_hirarchical(df):
    # De-duplicate input data
    df = de_duplication_hierarchical(df)
   
    # Remove noise in input data
    df = noise_remover(df)
    # Translate data to English
    # df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df

def get_embeddings(df: pd.DataFrame):
    X = get_tfidf_embd(df)  # Get tf-idf embeddings
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)

def perform_modelling(data: Data, df: pd.DataFrame, name):
    model_predict(data, df, name)

if __name__ == '__main__':
    df = load_data()
    #df_hierarchical = load_data_for_hierarchical()
    #df_hierarchical=preprocess_data_hirarchical(df_hierarchical)
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    grouped_df = df.groupby(Config.GROUPED)
    for name, group_df in grouped_df:
        print(name)
        X, group_df = get_embeddings(group_df)
        data = get_data_object(X, group_df)
        perform_modelling(data, group_df, name)
        
        # Define true classification for the email instance
        true_classification = {
            'Type 2': 'Suggestion',
            'Type 3': 'Payment',
            'Type 4': 'Subscription cancellation'
        }

        # Define predictions for scenarios B and C
        predictions_B = {
            'Type 2': 'Suggestion',
            'Type 3': 'Payment',
            'Type 4': 'Subscription retained'
        }

        predictions_C = {
            'Type 2': 'Suggestion',
            'Type 3': 'Refund',
            'Type 4': 'Subscription retained'
        }

        # Calculate predictions for Scenario A
        predictions_A = true_classification
        # Print predictions and accuracies
        print("Predictions for Scenario A:", predictions_A)
       
        # Calculating accuracies for Scenario B and C
        accuracy_B = calculate_accuracy(true_classification, predictions_B)
        accuracy_C = calculate_accuracy(true_classification, predictions_C)
        print("Accuracy for Scenario A:", calculate_accuracy(true_classification, predictions_A), "%")
        print("Accuracy for Scenario B:", accuracy_B, "%")
        print("Accuracy for Scenario C:", accuracy_C, "%")



        # Example usage of HierarchicalModel
        #hierarchical_model = HierarchicalModel(df)
        #new_instance_type_2 = 'Suggestion'
        #new_instance_type_2 = predictions_A  # Example class from Type 2
        #predictions = hierarchical_model.predict(new_instance_type_2)
        #print("Predictions for Type 3 and Type 4:")
        #if predictions is not None:
           # for class_type_3, predicted_type_4 in predictions.items():
               # print("Type 3:", class_type_3, "Predicted Type 4:", predicted_type_4)
