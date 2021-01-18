import pandas as pd
import numpy as np

df = pd.read_csv('NSE-TATAGLOBAL11.csv')
close=df['Close']
df['Date']=pd.to_datetime(df['Date'])
df['year']=df['Date'].dt.year


cls1=df[df.year<2017]
cls2=df[df.year==2017]
cls3=df[df.year>=2018]

print(len(cls2.Date))

close1=df[df.year<2017].Close
close2=df[df.year==2017].Close
close3=df[df.year>=2018].Close

import matplotlib.pyplot as plt

#plot1
plt.figure(figsize=(5,5))
plt.scatter(cls1.Date,close1)
plt.xlabel('date')
plt.ylabel('close')
plt.title('Dataset 1')
plt.show()


#plot2
plt.figure(figsize=(5,5))
plt.scatter(cls2.Date,close2)
plt.xlabel('date')
plt.ylabel('close')
plt.title('Dataset 2')
plt.show()


#plot3
plt.figure(figsize=(5,5))
plt.scatter(cls3.Date,close3)
plt.xlabel('date')
plt.ylabel('close')
plt.title('Dataset 3')
plt.show()


#plot complete
plt.figure(figsize=(5,5))
plt.scatter(df['Date'],close)
plt.xlabel('date')
plt.ylabel('close')
plt.title('Dataset(Complete)')
plt.show()

#<2017
# setting the index as date
cls1['Date'] = pd.to_datetime(cls1.Date,format='%Y-%m-%d')
cls1.index = cls1['Date']

data = cls1.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(cls1)),columns=['Date', 'Close'])

for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['Close'][i] = data['Close'][i]

# splitting into train and validation
train = new_data[:650]
valid = new_data[650:]

preds = []
for i in range(0,valid.shape[0]):
    a = train['Close'][len(train)-147+i:].sum() + sum(preds)
    b = a/147
    preds.append(b)

rms1=np.sqrt(np.mean(np.power((np.array(valid['Close'])-preds),2)))
print('\n RMSE value on validation set:')
print(rms1)

#between 2017-2018

cls2['Date'] = pd.to_datetime(cls2.Date,format='%Y-%m-%d')
cls2.index = cls2['Date']

data = cls2.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(cls2)),columns=['Date', 'Close'])

for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['Close'][i] = data['Close'][i]

# splitting into train and validation
train = new_data[:240]
valid = new_data[240:]

preds = []
for i in range(0,valid.shape[0]):
    a = train['Close'][len(train)-8+i:].sum() + sum(preds)
    b = a/8
    preds.append(b)

rms2=np.sqrt(np.mean(np.power((np.array(valid['Close'])-preds),2)))
print('\n RMSE value on validation set:')
print(rms2)


#>2018

cls3['Date'] = pd.to_datetime(cls3.Date,format='%Y-%m-%d')
cls3.index = cls3['Date']

data = cls3.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(cls3)),columns=['Date', 'Close'])

for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['Close'][i] = data['Close'][i]

# splitting into train and validation
train = new_data[:160]
valid = new_data[160:]

preds = []
for i in range(0,valid.shape[0]):
    a = train['Close'][len(train)-30+i:].sum() + sum(preds)
    b = a/30
    preds.append(b)

rms3=np.sqrt(np.mean(np.power((np.array(valid['Close'])-preds),2)))
print('\n RMSE value on validation set:')
print(rms3)


#calculating value
print("\mAverage RMSE value of dataset")
print((rms1+rms2+rms3)/3)