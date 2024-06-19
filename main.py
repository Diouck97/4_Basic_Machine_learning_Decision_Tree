import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor #library to create models
from sklearn.model_selection import train_test_split #split data to train and test

file_path = "C:/Users/diouc/OneDrive/Bureau/Business/Datasets/Home Data for ML course/train.csv"
Iowa_home_data = pd.read_csv(file_path)

'''' print(Iowa_home_data.describe())
print(Iowa_home_data.columns) '''

y = Iowa_home_data.SalePrice #prediction target is the Sale Price
Iowa_Features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]  #Selecting a list of features for the prediction model
X = Iowa_home_data[Iowa_Features]

#print(X.describe())
#print(X.head())
#Iowa_model = DecisionTreeRegressor(random_state = 1)  #random_state parameter determines the random number generation used to create the decision tree. When you set a specific random_state, the decision tree will be generated using the same random number generation each time, which can be useful for reproducibility.
#Iowa_model.fit(X,y) #fit the models = alimenter
#predictions = Iowa_model.predict(X.head())

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X, train_y)

predictions = iowa_model.predict(val_X)

'''print("Making predictions for the following 5 houses:")
print(train_X.head())
print("The predictions are")
print(predictions)

MAE = mean_absolute_error(val_y, predictions)
print(MAE)'''

#this function returns the MAE automatically
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# We are gonna compare MAE from different tree sizes
'''for max_leaf_nodes in [5, 25, 50, 100, 250, 500]:
    MAE = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf node = %d \t\t Mean Absolute Error = %d" % (max_leaf_nodes, MAE))'''

max_leaf_nodes = [5, 25, 50, 100, 250, 500]

#using a dictionnary to automatically compare and detect best tree size
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)
print(best_tree_size)

#now that wee have the best tree size we automatically fit the model using the argument
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size,random_state=1 )
final_model.fit(X, y)
