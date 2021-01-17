import numpy as np 
import pandas as pd 
import sqlite3
from impactRecoveryMet import ImpactRecoveryMetrics

def ClassifyCurves(polyDict:dict,metrics:pd.DataFrame):
    
    if ~np.isnan(metrics.preconT[0]):
        metrics["c1"] = 1
    elif polyDict["yHat"].mean() < 0:
        metrics["c1"] = 2
    elif polyDict["yHat"].mean() > 0:
        metrics["c1"] = 3

    return metrics


if __name__ == "__main__":
    import sqlite3
    conn = sqlite3.connect("D:/GEE_Project/Databases/database.db")
    cur = conn.cursor()
    # Select AnomalyM
    q1 = "SELECT * FROM anomalyM"
    anomalyM = pd.read_sql(sql = q1, con = conn) 

    #select AnomalyY
    q2 = "SELECT * FROM anomalyY"
    anomalyY = pd.read_sql(sql = q2, con = conn)

    #select Points
    q3 = "SELECT * FROM points"
    points = pd.read_sql(sql = q3, con = conn)

    # Join points and anomalyM
    anomPts = pd.merge(points, anomalyM, on = "id")

    # Filter out values eruption cc2011
    cc = anomPts.query('eruption == "CC2011"')
    
    uniqIds = cc.id.unique()
    ccId = cc[cc.id == uniqIds[9]]

    print(ccId["LSSR.EVI.CDI"].isna().any())
    
    # Call Impact Recovery Metrics class
    inst = ImpactRecoveryMetrics(data = ccId, target = "LSSR.EVI.CDI",eDate = "2011-06-04")
    polyDict = inst.fitPolyFunc()
    met = inst.getScore()

    # Call classify curve function
    metU= ClassifyCurves(polyDict = polyDict,metrics = met)

    # To visualize
    #fig = inst.visFunc()
    #fig.show()

    # -------------------------------------------------------------------------------------------
    # To compute on all uniqIds
    metList = []
    for id in uniqIds:
        ccId = cc[cc.id == id]
        inst = ImpactRecoveryMetrics(data = ccId, target = "LSSR.EVI.CDI",eDate = "2011-06-04")
        polyD = inst.fitPolyFunc()
        met = inst.getScore()
        metU = ClassifyCurves(polyDict = polyD, metrics = met)

        metList.append(metU)
        metAll = pd.concat(metList)

        print(metAll)
