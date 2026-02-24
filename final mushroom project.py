import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn import metrics  # The math part
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
# when you want to look through the whole data frame
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

dfmain = pd.read_csv("dataset/agaricus-lepiota.data", header=None)
plt.savefig("images/heatmap.png")
#Class (y):

# edible: 4208 (51.8%) --> 'e'
# poisonous: 3916 (48.2%) --> 'p'

# Manually create dictionary
attr = {
    'cap-shape':                ['bell=b,conical=c,convex=x,flat=f,knobbed=k,sunken=s'],
    'cap-surface':              ['fibrous=f,grooves=g,scaly=y,smooth=s'],
    'cap-color':                ['brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y'],
    'bruises':                  ['bruises=t,no=f'],
    'odor':                     ['almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s'],
    'gill-attachment':          ['attached=a,descending=d,free=f,notched=n'],
    'gill-spacing':             ['close=c,crowded=w,distant=d'],
    'gill-size':                ['broad=b,narrow=n'],
    'gill-color':               ['black=k,brown=n,buff=b,chocolate=h,gray=g,green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y'],
    'stalk-shape':              ['enlarging=e,tapering=t'],
    'stalk-root':               ['bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?'],
    'stalk-surface-above-ring': ['fibrous=f,scaly=y,silky=k,smooth=s'],
    'stalk-surface-below-ring': ['fibrous=f,scaly=y,silky=k,smooth=s'],
    'stalk-color-above-ring':   ['brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y'],
    'stalk-color-below-ring':   ['brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y'],
    'veil-type':                ['partial=p,universal=u'],
    'veil-color':               ['brown=n,orange=o,white=w,yellow=y'],
    'ring-number':              ['none=n,one=o,two=t'],
    'ring-type':                ['cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z'],
    'spore-print-color':        ['black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y'],
    'population':               ['abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y'],
    'habitat':                  ['grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d'],
}

print("Number of feaures:", len(attr))

# Goes through the array seperated by ',' reads string
for key, value in attr.items():
    attr[key]= value[0].split(',')

# Read the data file
dfmain = pd.read_csv(r'C:\Users\setayesh\Desktop\mush\agaricus-lepiota.data', header=None)
print(dfmain.shape)
print(dfmain.head(2))

# Add column names
print(dfmain.columns)
dfmain.columns=['class']+list(attr.keys())
print(dfmain)


# Check for missing values
print(dfmain.isna().sum())


# Split training features and output class
# first looking at all the columns in dfmain that arent "class"
Xmain = dfmain[dfmain.columns[~dfmain.columns.isin(['class'])]]
y = dfmain['class']


# #If yoou just want to get the names of columns wihtout showing the whole thing
# # print(Xmain.columns)


# # Check the proportion of each class
# # print(dfmain.groupby(['class'])['class'].count())


#### PREPROCCESSING
# Creating dummie variables
Xmain=pd.get_dummies(Xmain)
print("dummies :::::: ",Xmain.columns)
print(Xmain.head(2))  # if that feature was used 1 else 0


### CHECK FOR COLLINEARITY
# if there are features that do the same thing you want to remove them
x_corr = Xmain.corr()**2
print(x_corr)
# print("All feature labels may not be visible on the plot")
fig = plt.figure(figsize=(20,20))
sns.heatmap(x_corr.round(2)) #, annot=True)
plt.savefig(r'C:\Users\setayesh\Desktop\heatmap.png')
#### Remove correlated columns or features
def remove_colinear_cols(X):
    cols = list(X.columns)
    print("Numer of features (before):", len(cols))
    
    for col in cols:
        for icol in cols:
            if(col!=icol):
                rsq = np.corrcoef(X[col], X[icol])[0,1]**2
                if((rsq >=0.7) | (rsq <= -0.7)):
                    cols.remove(icol)
    print('Number of features (after):', len(cols))
    return cols
# # Update keep only non-colinear features
new_cols = remove_colinear_cols(Xmain)


# # With colinearity
X = Xmain.copy()

# # Removed colinearity
X = Xmain[new_cols]
print(X.shape)
# # Removing the white line in the heatmap
X = X[X.columns[~X.columns.isin(['veil-type_p'])]]
print(X.shape)
#### CHECK HEATMAP AGAIN
x_corr = X.corr()**2
print("All feature labels may not be visible on the plot")
fig = plt.figure(figsize=(20,20))
sns.heatmap(x_corr.round(2)) #, annot=True)
plt.savefig(r'C:\Users\setayesh\Desktop\heatmap2.png')


# #### SPLIT DATA
X=np.array(X)
x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=100)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#6093+2031
print(dfmain.shape)








clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=3, min_samples_leaf=5)
#max-depth means its only gonna go down three layers(no more than 3 splits)
#at least 5 leaes at the end
clf_entropy.fit(x_train,y_train)
print(y_train)

# TESTING PREDICTION
y_pred_en=clf_entropy.predict(x_test)
print(y_pred_en)
print ("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)
# Confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred_en)
print("Confusion matrix:\n", cm)

print(y)