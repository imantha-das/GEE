# Imports 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np 
import pandas as pd 
import sqlite3 
from kmodes.kprototypes import KPrototypes
import plotly.express as px
from termcolor import colored

# Load Data 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def connect2db(path:str, tb:str):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    q = "SELECT * FROM" + " " + tb 
    df = pd.read_sql(sql = q, con = conn)
    conn.close()

    return df

# Get points and metrics tables
points = connect2db(path = "D:/GEE_Project/Databases/database.db", tb = "points")
metrics = connect2db(path = "D:/GEE_Project/Databases/polyfitsMets.db", tb = "metrics")

# Filter eruption & Dataset 
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#Join tables
ptsMet = pd.merge(points, metrics , on = "id")

def getEruptionData(data: pd.DataFrame, eruption:str = "Merapi2010", dataset:str = "LSSR.NDVI.CDI"):
    data = data[(data.eruption == eruption)&(data.dataset == dataset)]
    return data

erupData = getEruptionData(data = ptsMet)
print(colored(erupData.shape,"red"))

#drop missing values 
#erupData.dropna(inplace = True)


# K-Prototype clustering : Feature Processing
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def kprototypesCluster(features:np.array, catCols:list, nClust:int):

    #Convert continous features to astype float
    model = KPrototypes(n_clusters = nClust, verbose = 2)
    clusters = model.fit_predict(features, categorical = catCols)
    
    return model

# drop columns that re not required : id, lat, lon, eruption, dataset, delayT, delayTS, delayV, preconT, preconTS, preconV, improvDeclT, improvDeclTS

dropColLs = [word for word in erupData.columns.values if word.endswith(("T","TS","V"))]  
for i in [["id","lat","lon","eruption","dataset"]]:
    dropColLs.extend(i)

# Remove preconV from dropColLs
dropColLs.remove("preconV")

# Since preconV is considered all point which didnot reach precondition has been dropped
df = erupData.drop(dropColLs, axis = 1)
df.dropna(inplace = True)
y = df.preconV

#drop preconV and select Features
df.drop("preconV", inplace = True, axis =1)
X = df.values

# Convert continous features float64 vals
df.elevation = df.elevation.astype("float64")

#Categorical Columns : label - 0, CHILI - 6, landform - 7, landcoverGFSAD - 8, landcoverGLOBECOVER - 9, climate - 10, Soiltype - 11, ;landcoverCGLS - 15, MODISLC - 16 
catCols = [0,6,7,8,9,10,11,15,16]

print(colored(X.shape, "red"))

# Run K-Prototype Clustering 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""
clusCost = list()
# Call kprototype
for n in np.arange(1,9):
    print(colored(n,"red"))
    clus = kprototypesCluster(features = X, catCols = catCols, nClust = n)
    clusCost.append(clus.cost_)

print(colored(clusCost[:8],"red"))

# Elbow plot
fig = px.line(
    x = np.arange(1,9),
    y = clusCost[:8]
)

fig.show()
"""

# Select 3 as the number of clusters
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
model3 = kprototypesCluster(features = X, catCols = catCols, nClust = 3)

df["clustLabels"] = model3.labels_
print(df.head())
