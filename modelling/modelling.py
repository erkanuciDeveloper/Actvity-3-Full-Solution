from model.randomforest import RandomForest
from model.decisiontree import DecisionTree

def model_predict(data, df, name):
    results = []
    print("RandomForest Model Result Information")
    model = RandomForest("randomforest", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)
 
 #----------------------------------------
    print("Decision Tree Model Result Information")  # Changed the print statement to reflect the model being used
    model = DecisionTree("decisiontree", data.get_embeddings(), data.get_type())  # Changed to DecisionTree
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)




def model_evaluate(model, data):
    model.print_results(data)
