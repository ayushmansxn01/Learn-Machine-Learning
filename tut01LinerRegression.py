import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
 # sk learn provides buildin data sets sets like uci machine learning repository
from sklearn.metrics import mean_squared_error


diabetes= datasets.load_diabetes()   # to import diabetes data set from sk learn pre existing data sets
print(diabetes.keys())         # to find the data inputs in diabetes data set
#dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
#print(diabetes.DESCR)


#diabetes_x=diabetes.data[:, np.newaxis,2] #feature at index 2 
diabetes_x=diabetes.data

diabetes_x_train=diabetes_x[:-30] # take last 30 for training
diabetes_x_test=diabetes_x[-30:]

diabetes_y_train=diabetes.target[:-30 ]
diabetes_y_test=diabetes.target[-30:]

model= linear_model.LinearRegression()

model.fit(diabetes_x_train, diabetes_y_train)

diabetes_y_predicted=model.predict(diabetes_x_test)

print("mean square error is :", mean_squared_error(diabetes_y_test, diabetes_y_predicted))
print("weights: ",model.coef_ )
print("intercept: ", model.intercept_ )


# TO PLOT THE GRAPH

# plt.scatter(diabetes_x_test, diabetes_y_test)
# plt.plot(diabetes_x_test, diabetes_y_predicted)
# plt.show()


#ERROR WITH ONLY ONE VALUE
# mean square error is : 2520.2122548902375
# weights:  [945.4992184]
# intercept:  152.33489819153206
