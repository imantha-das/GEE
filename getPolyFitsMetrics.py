# Function to retrieve polynomial fits data and impact revovery metrics
# ----------------------------------------------------------------------------------------------------------
def getFitsScore(df, eruptionName, eruptionDate):

    assert "eruption" in df.columns.values, "Eruption column missing in data"
    
    data = df[df.eruption == eruptionName]
    uniqIds = data.id.unique()

    targets = [i for i in data.columns.values if (i.startswith(("MODIS","LSSR"))) & (i.endswith("CDI"))]

    metricsList = []
    polyFitList = []
    
    # for loop begins
    for i in uniqIds:
        tempPolyList = []
        polyDf = pd.DataFrame()
        for j in targets:

            dataPerId = data[data.id == i]

            # Construct ImpactRecoveryMetrics instance
            inst = ImpactRecoveryMetrics(data = dataPerId, target = j, eDate = eruptionDate)

            # Compute and extract polynomial fitted values
            polyDf[j + "." + "FIT"] = inst.fitPolyFunc()["yHat"]
            polyDf["id"] = i
            tempPolyList.append(polyDf)

            # Compute metrics
            metricsDf = inst.getScore()
            metricsDf["dataset"] = j

            metricsList.append(metricsDf)

        tempPolyDf = pd.concat(tempPolyList, axis = 1)
        polyFitList.append(tempPolyDf)

    metrics = pd.concat(metricsList)
    #Move dataset to column 1
    firstCol = metrics.pop("dataset")
    metrics.insert(1,"dataset", firstCol)

    polyfits = pd.concat(polyFitList)
    # Remove duplicate id columns
    polyfits = polyfits.loc[:, ~polyfits.columns.duplicated()]
    #Move id to column 0
    zeroCol = polyfits.pop("id")
    polyfits.insert(0, "id", zeroCol)

    return polyfits,metrics


# Function to write data to database
# -----------------------------------------------------------------------------------------------------

def writeToDB(path:str,dbName:str,tbName:str, df:pd.DataFrame):
    import sqlite3

    conn = sqlite3.connect(path + "/" + dbName)
    cur = conn.cursor()

    # Drop table if exists
    q1 = "drop TABLE if EXISTS" + " " + tbName
    cur.execute(q1)

    # Constructs a new table
    q2 = "CREATE TABLE" + " " + tbName + " " + "(id text, dataset text, delayT number, dalayTS text delayV number, preconT number, preconTS text, preconV number, improvDeclT number, improvDeclTS text, improvDeclV number, budgetT number, budgetTS text, budgetV number)"
    
    cur.execute(q2)

    # Write pandas dataframe
    #Convert to pandas datetime object
    colNameTS = df.columns.values[[i.endswith("TS") for i in df.columns.values]]

    for nm in colNameTS:
        df[nm] = df[nm].dt.strftime("%Y-%m-%d")

    df.to_sql(tbName, conn, if_exists="replace", index = False)

if __name__ == "__main__":

    polyfits,metrics = getFitsScore(df = anomPts, eruptionName = "Merapi2010", eruptionDate = "2010-10-26")

    
