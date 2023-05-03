
#importing Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:

#importing dataset
df=pd.read_csv(r"C:\Users\asus\Data_Analytics\heart_failure_clinical_records_dataset.csv")


# In[3]:

#checking the dataset
df


# In[4]:

#checking unique values in the dataset
df.nunique()


# In[5]:

#checking the info of the statset
df.info()


# In[6]:

#checking for null values
df.isnull().any()


# In[7]:
#checking the shape of the dataset

df.shape


# In[8]:

#checking for duplicate values
df.duplicated().sum()


# In[9]:

#checking the description of the dataset
df.describe(include="all").transpose()


# ### EDA/Visualization

# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


#plotting a correlation matrics for the dataset to display the correlation cooeficients between pairs in the dataset 
corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap='viridis', annot=True, fmt='.2f')
plt.show()


# In[12]:


sns.displot(df['platelets'])


# Outliers observed 

# In[13]:


sns.displot(df['age'])


# In[14]:


sns.displot(df['DEATH_EVENT'])


# In[ ]:





# ### Model Building

# ### KNeighborsClassifier

# In[15]:


X=df.iloc[:,0:12]
y=df.iloc[:,-1]


# In[16]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(X)



# In[17]:


x


# In[18]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)


# In[19]:


# Creating Function
def predict(model):
    model.fit(X_train,y_train)
    ypred=model.predict(X_test)
    traindf=model.score(X_train,y_train)
    testdf=model.score(X_test,y_test)
    
    print(f"Triaing Accuracy {traindf}\nTesting Accuracy {testdf}")
    


# In[20]:


from sklearn.neighbors import KNeighborsClassifier

# KNN
model_knn=KNeighborsClassifier(n_neighbors=6)
model_knn.fit(X_train,y_train)
ypred_knn=model_knn.predict(X_test)
traindf_knn=model_knn.score(X_train,y_train)
testdf_knn=model_knn.score(X_test,y_test)
print(f"Triaing Accuracy {traindf_knn}\nTesting Accuracy {testdf_knn}")  


# In[21]:


traindf=[]
testdf=[]

for i in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    ypred=knn.predict(X_train)
    
    traindf.append(knn.score(X_train,y_train))
    testdf.append(knn.score(X_test,y_test))


# In[22]:


sns.set_style(style='darkgrid')

plt.plot(range(1,20), traindf)
plt.plot(range(1,20),testdf)
plt.xlabel('Number of K')
plt.ylabel('Accuracy')


# ### Linear Regression

# In[23]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[24]:


lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[25]:


# Predict the target variable on the test data
y_pred = lr.predict(X_test)


# In[26]:


y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)


# In[27]:


train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)



# In[28]:


# Print the accuracy scores
print("Training accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)


# ### RandomForestClassifier

# In[29]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[30]:


rf = RandomForestClassifier(n_estimators=100)


# In[31]:


rf.fit(X_train, y_train)


# In[32]:


y_pred = rf.predict(X_test)


# In[33]:


train_score = rf.score(X_train, y_train)
test_score = accuracy_score(y_test, y_pred)
print("Training Acuracy:", train_score)
print("Test Acuracy:", test_score)


# In[34]:


fig, ax = plt.subplots()
ax.scatter(np.arange(len(y_test)), y_test, color='blue', label='Actual')
ax.scatter(np.arange(len(y_pred)), y_pred, color='red', label='Predicted')
ax.legend(loc='best')
ax.set_xlabel('Sample index')
ax.set_ylabel('Target value')
ax.set_title('Actual vs Predicted Values')
plt.show()


# ### comparing the training and test acuracy for the three models

# In[35]:


import warnings
warnings.filterwarnings("ignore")



import pandas as pd

results_df = pd.DataFrame(columns=['Model', 'Training Accuracy', 'Testing Accuracy'])

results_df = results_df.append({'Model': 'KNeighborsClassifier', 
                                'Training Accuracy': 0.772, 
                                'Testing Accuracy': 0.773}, ignore_index=True)

results_df = results_df.append({'Model': 'RandomForestClassifier', 
                                'Training Accuracy': 1.0, 
                                'Testing Accuracy': 0.92}, ignore_index=True)

results_df = results_df.append({'Model': 'Linear Regression', 
                                'Training Accuracy': 0.83, 
                                'Testing Accuracy': 0.867}, ignore_index=True)



results_df = results_df.append({'Model': 'LSTM', 
                                'Training Accuracy': 0.728, 
                                'Testing Accuracy': 0.6}, ignore_index=True)

print(results_df)



# ### Stastmodels for prediction 

# ### Arima 

# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# In[37]:


df=pd.read_csv(r"C:\Users\asus\Data_Analytics\heart_failure_clinical_records_dataset.csv")
df


# In[38]:


# Extract variables
y = df["DEATH_EVENT"]
X = df.drop("DEATH_EVENT", axis=1)


# In[39]:


#the pacf and acf and their confidence interval
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import statsmodels.api as sm
from statsmodels.tsa.stattools import pacf
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
sm.graphics.tsa.plot_pacf(y, lags=10, ax=ax[0])
ax[0].set(title="Partial Autocorrelation")
sm.graphics.tsa.plot_acf(y, lags=10, ax=ax[1])


# In[40]:


train_size = int(len(df) * 0.7)
train_data, test_data = df[:train_size], df[train_size:]


# In[41]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima
# Use auto_arima to find the best parameters for the ARIMA model
stepwise_model = auto_arima(train_data['DEATH_EVENT'], start_p=0, start_q=0,
                            max_p=5, max_q=5, m=1, start_P=0, seasonal=False,
                            d=1, D=1, trace=True, error_action='ignore',
                            suppress_warnings=True, stepwise=True)


# In[42]:


model = ARIMA(train_data['DEATH_EVENT'], order=(0,1,1))
model_fit = model.fit()


# In[43]:


print(model_fit.summary())


# In[44]:


predictions = model_fit.predict(start=len(train_data), end=len(df)-1, typ='levels')
predictions


# In[45]:


# Evaluate the model performance
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
mse = mean_squared_error(test_data['DEATH_EVENT'], predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_data['DEATH_EVENT'], predictions)
msle = mean_squared_log_error(test_data['DEATH_EVENT'], predictions)
print('MSE:', mse)
print('RMSE:', rmse)
print('MAE:', mae)
print('MSLE:', msle)


# In[46]:


# Plot the actual and predicted values
plt.plot(test_data.index, test_data['DEATH_EVENT'], label='Actual')
plt.plot(test_data.index, predictions, label='Predicted')
plt.title('Actual vs Predicted Values')
plt.xlabel('Time')
plt.ylabel('DEATH_EVENT')
plt.legend()
plt.show()


# ### Sarimax Model

# In[47]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[48]:


# Fit the SARIMAX model
model = SARIMAX(train_data['DEATH_EVENT'], order=(1, 0, 0), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()


# In[49]:


# Generate predictions for the test set
predictions = model_fit.predict(start=len(train_data), end=len(df)-1, typ='levels')


# In[50]:


# Evaluate the model's performance using MSE, RMSLE, MSLE, and RMSE
mse = mean_squared_error(test_data['DEATH_EVENT'], predictions)
rmsle = np.sqrt(mean_squared_log_error(test_data['DEATH_EVENT'], predictions))
msle = mean_squared_log_error(test_data['DEATH_EVENT'], predictions)
rmse = np.sqrt(mse)
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('Root Mean Squared Log Error:', rmsle)
print('Mean Squared Log Error:', msle)


# In[51]:


# Plot the predicted values against the actual values
plt.plot(test_data.index, test_data['DEATH_EVENT'], label='Actual')
plt.plot(predictions.index, predictions, label='Predicted')
plt.title('Actual vs Predicted Values')
plt.xlabel('Time')
plt.ylabel('DEATH_EVENT')
plt.legend()
plt.show()


# ### Comparing both the Arima and Sarimax model

# In[52]:


import pandas as pd

# create empty dataframe
results_df = pd.DataFrame(columns=['Model', 'MSE', 'RMSE', 'MAE', 'MSLE'])

# append values for ARIMA model
results_df = results_df.append({'Model': 'ARIMA', 'MSE': 0.0717615966388786, 'RMSE': 0.2678835505193975, 'MAE': 0.13859103285657762, 'MSLE': 0.034706397287345335}, ignore_index=True)

# append values for SARIMAX model
results_df = results_df.append({'Model': 'SARIMAX', 'MSE': 0.10564954267485367, 'RMSE': 0.32503775576823946, 'MAE': None, 'MSLE': 0.06322133876961758}, ignore_index=True)

print(results_df)


# ### LSTM Model

# In[53]:


df=pd.read_csv(r"C:\Users\asus\Data_Analytics\heart_failure_clinical_records_dataset.csv")
df


# In[54]:


# Split the data into features (X) and target (y)
X = df.drop(['DEATH_EVENT'], axis=1)
y = df['DEATH_EVENT']


# In[55]:


# Scale the data using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# In[56]:


# Reshape the data into 3D format expected by LSTM (samples, timesteps, features)
import numpy as np
X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))


# In[57]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


# In[58]:


# Create the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(1, X.shape[1]), activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[59]:


# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[60]:


# Fit the model to the data
model.fit(X_reshaped, y, epochs=50, batch_size=32, validation_split=0.2)


# In[ ]:




