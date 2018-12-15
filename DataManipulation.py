# Node Flooding example
# This Python code was used to prepare my ANN data from Homework #1 for use
# by ABAGAIL in Homework #2

# Import the necessary libraries
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Set the working directory (set to directory containing the dataset)
os.chdir('C:\\Users\Brad\Desktop\Briefcase\Personal\GeorgiaTechMasters\CS7641_MachineLearning\Homework\Homework1')

# Import the dataset and collect some high-level information on it
dataset = pd.read_csv('NodeFlooding.csv')
print ("Dataset Length = ", len(dataset))
print ("Dataset Shape = ", dataset.shape)
print (dataset.head())

# Separate out the pertinent independent variables from the dataset
ind_var = dataset.iloc[:, [19, 1, 2, 4, 6, 7, 16, 17, 18, 20]].values
print(ind_var[:10, :])

# Encode the categorical data
# Encode the Independent variables
labelencoder_ind = LabelEncoder()
ind_var[:, 0] = labelencoder_ind.fit_transform(ind_var[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
ind_var = onehotencoder.fit_transform(ind_var).toarray()
Ind_Headers = np.array(['NodeStatus_B', 'NodeStatus_NB', 'NodeStatus_PNB',
                      'UtilBandwRate', 'PacketDropRate', 'AvgDelay_perSec',
                      'PctLostByteRate', 'PacketRecRate', '10RunAvgDropRate',
                      '10RunAvgBandwUse', '10RunDelay', 'FloodStatus'])
print(Ind_Headers)
print(ind_var[:10, :])

# Separate out the pertinent dependent variables from the dataset
dep_var = dataset.iloc[:, 21].values
print(dep_var[:10])

# Encode the categorical data
# Encode the Dependent variables
labelencoder_dep = LabelEncoder()
dep_var = labelencoder_dep.fit_transform(dep_var)
Dep_Results = np.array(['0=Block', '1=NB-No_Block', '2=NB-Wait', '3=No_Block'])
print(Dep_Results)
print(dep_var[:10])

# Scale the independent variables for ease of ANN calculation
scaler = StandardScaler()
scaler.fit(ind_var)
ind_var_scaled = scaler.transform(ind_var)

# Concatenate the dependent variable (classifier) onto the independent variables
dep_var = dep_var.reshape(dep_var.shape[0], -1)
flood_data = np.concatenate((ind_var_scaled, dep_var), 1)
print(flood_data[:10])

# Binary classifier - Block = 0, Other = 1
flood_data[:, 12][flood_data[:, 12] > 0] = 1.0
print(flood_data[:10])

# Export as a csv file
df = pd.DataFrame(flood_data)
df.to_csv('out.csv', index=False, header=False)
# The csv file could then be moved to the appropriate ABAGAIL repository directory
# and leveraged in the pertinent ABAGAIL program


# Vehicle Evaluation example

# Import the necessary libraries
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Set the working directory (set to directory containing the dataset)
os.chdir('C:\\Users\Brad\Desktop\Briefcase\Personal\GeorgiaTechMasters\CS7641_MachineLearning\Homework\Homework1')

# Import the dataset and collect some high-level information on it
dataset = pd.read_csv('CarRatingDataset.csv')
print ("Dataset Length = ", len(dataset))
print ("Dataset Shape = ", dataset.shape)
print (dataset.head())

# Separate out the pertinent independent variables from the dataset
ind_var = dataset.iloc[:, :6].values
print(ind_var[:10, :])

# Encode the categorical data
# Encode the Independent variables
labelencoder_ind = LabelEncoder()
ind_var[:, 0] = labelencoder_ind.fit_transform(ind_var[:, 0])
ind_var[:, 1] = labelencoder_ind.fit_transform(ind_var[:, 1])
ind_var[:, 2] = labelencoder_ind.fit_transform(ind_var[:, 2])
ind_var[:, 3] = labelencoder_ind.fit_transform(ind_var[:, 3])
ind_var[:, 4] = labelencoder_ind.fit_transform(ind_var[:, 4])
ind_var[:, 5] = labelencoder_ind.fit_transform(ind_var[:, 5])
onehotencoder = OneHotEncoder(categorical_features = [0, 1, 2, 3, 4, 5])
ind_var = onehotencoder.fit_transform(ind_var).toarray()
Ind_Headers = np.array(['BuyPrice_high', 'BuyPrice_low', 'BuyPrice_med',
                      'BuyPrice_vhigh', 'MaintPrice_high', 'MaintPrice_low',
                      'MaintPrice_med', 'MaintPrice_vhigh', '2-door', '3-door',
                      '4-door', '5more-door', '2-pass', '4-pass', '5more-pass',
                      'Luggage_big', 'Luggage_med', 'Luggage_small',
                      'safety_high', 'safety_low', 'safety_med'])
print(Ind_Headers)
print(ind_var[:10, :])

# Separate out the pertinent dependent variables from the dataset
dep_var = dataset.iloc[:, 6].values
print(dep_var[:10])

# Encode the categorical data
# Encode the Dependent variables
labelencoder_dep = LabelEncoder()
dep_var = labelencoder_dep.fit_transform(dep_var)
Dep_Results = np.array(['0=acc', '1=good', '2=unacc', '3=vgood'])
print(Dep_Results)
print(dep_var[:10])

# Scale the independent variables for ease of ANN calculation
scaler = StandardScaler()
scaler.fit(ind_var)
ind_var_scaled = scaler.transform(ind_var)

# Concatenate the dependent variable (classifier) onto the dependent variables
dep_var = dep_var.reshape(dep_var.shape[0], -1)
eval_data = np.concatenate((ind_var_scaled, dep_var), 1)
print(eval_data[:10])

# Binary classifier - Block = 0, Other = 1
eval_data[:, 21][eval_data[:, 21] == 0.0] = 1.0
eval_data[:, 21][eval_data[:, 21] == 2.0] = 0.0
eval_data[:, 21][eval_data[:, 21] == 3.0] = 1.0
print(eval_data[:10])

# Export as a csv file
df = pd.DataFrame(eval_data)
df.to_csv('out.csv', index=False, header=False)
