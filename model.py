import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  #for plotting
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

df=pd.read_csv("housing.csv")  #df is dataframe

#converting ocean proximity into nos 
#mapping = {
  #  "NEAR BAY":0,
   ## "INLAND":1,
    #"NEAR OCEAN":2,
    #"ISLAND":3
#}

#df["ocean_proximity"] = df["ocean_proximity"].map(mapping)
#df["ocean_proximity"] =df["ocean_proximity"].astype(float)

#one hot encoding
df = pd.get_dummies(df, columns=["ocean_proximity"])
print(df.head())  #prints the first 5 data of the dataset
print(df.info())  #shows the dataframe
print(df.isnull().sum())  #checking missing values

#filling missing values in empty rows using mean 
#fillna means replacing values with null values
df["total_bedrooms"]=df["total_bedrooms"].fillna(df["total_bedrooms"].mean())  #using inplace=True update then and there instead of writing df=

#feature scaling of median house value
#mean=df["median_house_value"].mean()
#std=df["median_house_value"].std()
#df["median_house_value"]=(df["median_house_value"]-mean)/std
#print(df["median_house_value"])
#
#mean1=df["population"].mean()
#std1=df["population"].std()
#df["population"]=(df["population"]-mean1)/std1

#plotting relationship between two features
plt.scatter(df["population"].head(100) , df["median_house_value"].head(100))  #in this fxn first one is x axis and second is y axis
plt.xlabel("populaiton")
plt.ylabel("value of house")
plt.show()

#scaling only the numeric ones 
scaler=StandardScaler()
numeric_cols = df.select_dtypes(include=["float64","int64"])
df[numeric_cols.columns] = scaler.fit_transform(numeric_cols)

#splitting the dataset

#here removing the output column from the input ones
X = df.drop("median_house_value", axis=1)  #axis=0 for rows axis=1 for column
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# convert to numeric
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# convert to tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1,1)

X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1,1)

class Mlp(torch.nn.Module):
    def __init__(self, num_features, num_hidden1=80, num_hidden2=120, num_hidden3=70, final=30, output=1):
      super().__init__()
      self.nu_net=torch.nn.Sequential(
         #Sequential is used for defining the order 

         torch.nn.Linear(num_features , num_hidden1),  #here it defines the size we cant initialize this
         torch.nn.BatchNorm1d(num_hidden1),
         torch.nn.Tanh(),  
        #hidden layer1
         torch.nn.Linear(num_hidden1 , num_hidden2),
         torch.nn.BatchNorm1d(num_hidden2),
         torch.nn.Tanh(),

        #hidden layer2
         torch.nn.Linear(num_hidden2 , num_hidden3),
         torch.nn.BatchNorm1d(num_hidden3),
         torch.nn.Tanh(),

        #hidden layer3
         torch.nn.Linear(num_hidden3 , final),
         torch.nn.BatchNorm1d(final),
         torch.nn.Tanh(),

         #output
         torch.nn.Linear(final , output),
      )

    def forward(self, x):
      return self.nu_net(x)
    
# device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

    #Initalising the neural network
num_features=X_train.shape[1]  #.shape will give a tuple(.. , ..)
                               #.shape[1] gives the second index and [0] the first index
model =Mlp(num_features)

#moving model to GPU
model=model.to(DEVICE)

#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# create mini-batch dataset
train_dataset = TensorDataset(X_train, y_train)  #Tensor dataset pairs them together  

train_loader = DataLoader(    #data loader splits the data into minibatches
    train_dataset,
    batch_size=32,
    shuffle=True
)

import time
start_time=time.time()
#training
epochs = 100

for epoch in range(epochs):

    model.train()
    epoch_loss = 0

    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        # forward pass
        predictions = model(X_batch)

        loss = torch.mean((predictions - y_batch)**2)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()  
        #loss.item() --> loss contains a pytorch tensor .item() is used to convert that tensor into a numerical value
        if batch_idx % 50 == 0:
            print("Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f"
                  % (epoch+1, epochs, batch_idx, len(train_loader), loss.item()))
            
    #average loss per epoch
    epoch_loss /= len(train_loader)
    print("Epoch: %03d/%03d Train Cost: %.4f"
          % (epoch+1, epochs, epoch_loss))
    print("Time elapsed: %.2f min"
          % ((time.time() - start_time) / 60))
    
# testing phase
model.eval()

#What model.eval does?
#it switches to evaluation mode
#instead of calculating it uses the stored value 

X_test = X_test.to(DEVICE)
y_test = y_test.to(DEVICE)

with torch.no_grad():  #this tells to not compute gradients

    test_predictions = model(X_test)  #forward pass through data

    test_loss = torch.mean((test_predictions - y_test) ** 2)

print("Test Loss:", test_loss.item())

#plotting 
actual = y_test.cpu().numpy()
predicted = test_predictions.cpu().numpy()

plt.figure(figsize=(8,6))

plt.scatter(actual, predicted, alpha=0.5)

plt.plot([actual.min(), actual.max()],
         [actual.min(), actual.max()],
         color="red")

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Actual vs Predicted House Prices")

plt.savefig("actual_vs_predicted.png")

print("Graph saved as actual_vs_predicted.png")

      


    
