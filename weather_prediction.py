from datetime import datetime
from meteostat import Point, Daily 
import numpy as np 
from sklearn.linear_model import LinearRegression


#set time period 
start = datetime(2018,1,1)
end = datetime.today() 

# Create Point for Waterloo, ON 
location = Point(43.4643, -80.5204, 329)

# Get Daily Data from 2018 
# start and end is datetime module 
data = Daily(location, start, end)
#pull the data
data = data.fetch() 

# # Plot line chart including average, minimum and maximum temperature
# data.plot(y=['tavg', 'tmin', 'tmax'])
# plt.show()

# Import data into numpy
# Parse data using the column names
train_data = np.array(data[['tavg', 'tmin', 'tmax']].values)
print(train_data.shape)

# Define sequence length in days (pre-processing)
n = 14
train_data_seq = []
for i in range(len(train_data)-n+1):
    train_data_seq.append(train_data[i:i+n])

# Now each sample has 14 days of data with 3 attributes 
# 1962 = 1975 - 14 + 1 
train_data_seq = np.array(train_data_seq)
print(train_data_seq.shape)

x = np.reshape(train_data_seq)
print(x)
#y = 
reg = LinearRegression().fit(train_data_seq, train_data)
reg.coef_