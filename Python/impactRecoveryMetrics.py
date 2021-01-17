import numpy as np
import pandas as pd 
import datetime
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import timedelta
import sys


class ImpactRecoveryMetrics(object):
    """Fits a Polynomial Function and computes the impact and recovery metrics from a time series signal"""
    
    def __init__(self,data:pd.DataFrame,target:str, eDate:str,deg = 4, timeThresh:int = 12):
        self.data = data
        self.target = target
        self.deg = deg
        self.eDate = eDate
        self.timeThresh = timeThresh * 31
        self.polyDict = dict()
        self.metDf = pd.DataFrame()
        self.isEmpty = False

        # Check if all values in target column is nan
        if self.data[self.target].isna().values.all():
            self.isEmpty = True
        #Ensure that only one unique Id is available in data
        assert data.id.unique().size == 1, "Data contain more than one unique ID"

        # convert time coloumn to datetime values
        try:
            data["time"] = pd.to_datetime(data["time"], format = "%Y-%m-%d")
            self.eDate = pd.to_datetime(eDate, format = "%Y-%m-%d")
        except:
            print("Time or Eruption date in incorrect format, Requires YYYY-MM-DD")
            sys.exit(1)


    def fitPolyFunc(self):
        """ Fits a polynomial function and cumputes it first and second derevatives"""

        if self.isEmpty == False:
            # Remove any missing values
            df = self.data[["time", self.target]]
            df.dropna(inplace = True)

            # Time since eruption : in days
            tmSinErup = (df.time - self.eDate)/np.timedelta64(1,"D") #Divide by np.timeDelta64 to get values in float64
            tmSinErup = np.array([int(i) for i in tmSinErup]) # Convert to integer values
            ndvi = df[self.target]
            
            # Fit data and construct a polynomial P
            p = np.polyfit(
                    x = tmSinErup,
                    y = ndvi,
                    deg = self.deg
                )

            # First and second dervatives of polynomial function p
            pDash1 = np.polyder(p)
            pDash2 = np.polyder(pDash1)

            # Predict values on original date, tmSinErup_o : original time series without dropped dates and ndvi
            tmSinErup_o = (self.data.time - self.eDate)/np.timedelta64(1,"D")
            tmSinErup_o = np.array([int(i) for i in tmSinErup_o])
            yHat = np.polyval(p,tmSinErup_o)

            # Store P
            self.polyDict["p"] = p
            self.polyDict["pDash1"] = pDash1
            self.polyDict["pDash2"] = pDash2
            self.polyDict["tmSinErup"] = tmSinErup_o
            self.polyDict["tmSinErupDt"] = self.data.time 
            self.polyDict["ndvi"] = ndvi 
            self.polyDict["yHat"] = yHat  

        else:
            self.polyDict["p"] = np.nan
            self.polyDict["pDash1"] = np.nan
            self.polyDict["pDash2"] = np.nan 
            self.polyDict["tmSinErup"] = np.empty(self.data.shape[0]) 
            self.polyDict["tmSinErupDt"] = np.empty(self.data.shape[0])
            self.polyDict["ndvi"] = np.empty(self.data.shape[0])
            self.polyDict["yHat"] = np.empty(self.data.shape[0])

            self.polyDict["tmSinErup"][:] = np.nan 
            self.polyDict["tmSinErupDt"][:] = np.nan 
            self.polyDict["ndvi"][:] = np.nan 
            self.polyDict["yHat"][:] = np.nan

        return self.polyDict

    def getScore(self):
        if self.isEmpty == False:
            # Compute roots of polynomial function and roots of first and second derevative --------------------
            # Compute roots of p (roots of f(x)) : x at y = 0
            pRtsAll = np.roots(self.polyDict["p"])
            pRtsReal = pRtsAll[np.isreal(pRtsAll)] # FIlter out complex roots
            pRts = pRtsReal[(pRtsReal >= self.polyDict["tmSinErup"][0]) & (pRtsReal <= self.polyDict["tmSinErup"][-1])] # filter out outside time range

            # Compute roots of first deravative : maxim and minima
            pDash1RtsAll = np.roots(self.polyDict["pDash1"])
            pDash1RtsReal = pDash1RtsAll[np.isreal(pDash1RtsAll)] #Filter 
            pDash1Rts = pDash1RtsReal[(pDash1RtsReal >= self.polyDict["tmSinErup"][0]) & (pDash1RtsReal <= self.polyDict["tmSinErup"][-1])] # filter out roots out side time range

            # Decide if its minima or a maxima : if pDsh2(x) > 0 - its a minima , if pDash2(x) < 0 - its a maxima
            maxminSign = np.polyval(self.polyDict["pDash2"],pDash1Rts) 

            # pDash1Rts Already filtered!!! : Hence dont need to filter minimaRts or maximaRts
            minimaRts = pDash1Rts[np.where(maxminSign > 0)]
            maximaRts = pDash1Rts[np.where(maxminSign < 0)]

            minimaVals = np.polyval(self.polyDict["p"], minimaRts)
            maximaVals = np.polyval(self.polyDict["p"], maximaRts)


            # Locate precondition point -------------------------------------------------------------------------
            # Rules : Must be on the negative side, if multiple values pick first
            preconIdx = np.where(minimaVals < 0)

            if preconIdx[0].size > 0:
                preconT = np.array([np.real(minimaRts[preconIdx][0])])
                preconTS = np.array([self.eDate + timedelta(days = preconT[0])])
                preconV = np.array([np.real(minimaVals[preconIdx][0])])
            else:
                preconT = np.array([np.nan])
                preconTS = np.array([np.nan])
                preconV = np.array([np.nan])

            # Locate Delay point -------------------------------------------------------------------------------
            # Rules : Must be before timeThresh = 12 months by default, must be on positive side
            delayIdx1 = np.where(maximaVals > 0) #must be on positive side
            delayIdx2 = np.where(maximaRts < self.timeThresh)
            delayIdx = np.intersect1d(delayIdx1,delayIdx2)

            if delayIdx.size > 0: 
                delayT = np.array([np.real(maximaRts[delayIdx][0])])
                delayTS = np.array([self.eDate + timedelta(days = delayT[0])])
                delayV = np.array([np.real(maximaVals[delayIdx][0])])
            else:
                delayT = np.array([np.nan])
                delayTS = np.array([np.nan])
                delayV = np.array([np.nan])

            # Locate Improvement Decline point -----------------------------------------------------------------
            #Rules : Must be after threshold, at the time not considering if precondition is necessary !!!
            improvDeclIdx = np.where(maximaRts > self.timeThresh)

            if improvDeclIdx[0].size > 0:
                improvDeclT = np.array([np.real(maximaRts[improvDeclIdx][0])])
                improvDeclTS = np.array([self.eDate + timedelta(days = improvDeclT[0])])
                improvDeclV = np.array([np.real(maximaVals[improvDeclIdx][0])])
            else:
                improvDeclT = np.array([np.nan])
                improvDeclTS = np.array([np.nan])
                improvDeclV = np.array([np.nan])

            # Locate budget point -----------------------------------------------------------------------------
            # Rules : Must be after threshold, must be after precon
            budgetIdx = np.where((pRts > self.timeThresh) & (pRts > preconT))

            if budgetIdx[0].size > 0:
                budgetT = np.array([np.real(pRts[budgetIdx][0])])
                budgetTS = np.array([self.eDate + timedelta(days = budgetT[0])])
                budgetV = np.array([0])
            else:

                budgetT = np.array([np.nan])
                budgetTS = np.array([np.nan])
                budgetV = np.array([np.nan])

            # Store all metrics in a dataframe ----------------------------------------------------------------
            self.metDf["id"] = self.data.id.unique()
            self.metDf["delayT"] = delayT
            self.metDf["delayTS"] = delayTS
            self.metDf["delayV"] = delayV
            self.metDf["preconT"] = preconT
            self.metDf["preconTS"] = preconTS
            self.metDf["preconV"] = preconV
            self.metDf["improvDeclT"] = improvDeclT
            self.metDf["improvDeclTS"] = improvDeclTS 
            self.metDf["improvDeclV"] = improvDeclV
            self.metDf["budgetT"] = budgetT
            self.metDf["budgetTS"] = budgetTS
            self.metDf["budgetV"] = budgetV

        else:
            self.metDf["id"] = self.data.id.unique()
            self.metDf["delayT"] = np.array([np.nan])
            self.metDf["delayTS"] = np.array([np.nan])
            self.metDf["delayV"] = np.array([np.nan])
            self.metDf["preconT"] = np.array([np.nan])
            self.metDf["preconTS"] = np.array([np.nan])
            self.metDf["preconV"] = np.array([np.nan])
            self.metDf["improvDeclT"] = np.array([np.nan])
            self.metDf["improvDeclTS"] = np.array([np.nan])
            self.metDf["improvDeclV"] = np.array([np.nan])
            self.metDf["budgetT"] = np.array([np.nan])
            self.metDf["budgetTS"] = np.array([np.nan])
            self.metDf["budgetV"] = np.array([np.nan])

        return self.metDf


    def visFunc(self):

        if self.isEmpty == False:
            #fig = make_subplots(rows = 2, cols = 1, subplot_titles = [str(self.metDf.id[0])])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x = self.data.time,
                y = self.data[self.target],
                mode = "lines",
                line = dict(
                    color = "dodgerblue",
                    width = 1,
                    dash = "dashdot"
                ),
                name = "anomaly actual"
            ))

            fig.add_trace(go.Scatter(
                x = self.data.time,
                y = self.polyDict["yHat"],
                mode = "lines",
                line = dict(
                    color = "slategray",
                    width = 3
                ),
                name = "anomaly fit"
            ))

            # Add delay trace
            fig.add_trace(go.Scatter(
                x = self.metDf.delayTS,
                y = self.metDf.delayV,
                mode = "markers",
                marker = dict(
                    color = "fuchsia",
                    size = 10
                ),
                name = "delay"
            ))

            # Add precondition Trace
            fig.add_trace(go.Scatter(
                x = self.metDf.preconTS,
                y = self.metDf.preconV,
                mode = "markers",
                marker = dict(
                    color = "coral",
                    size = 10
                ),
                name = "precondition"
            ))

            #Add Improvement Decline point Trace
            fig.add_trace(go.Scatter(
                x = self.metDf.improvDeclTS,
                y = self.metDf.improvDeclV,
                mode = "markers",
                marker = dict(
                    color = "limegreen",
                    size = 10
                ),
                name = "Improvement decline pt"
            ))

            # Add budget
            fig.add_trace(go.Scatter(
                x = self.metDf.budgetTS,
                y = self.metDf.budgetV,
                mode = "markers",
                marker = dict(
                    color = "steelblue",
                    size = 10
                ),
                name = "budget"
            ))

            fig.update_layout(
                xaxis_title = "Date", 
                yaxis_title = self.target, 
                title = self.metDf.id[0],
            )

            fig.update_xaxes(showgrid = False, zeroline = False)
            fig.update_yaxes(showgrid = False, zeroline = False)

            return fig

        else:
            Print("Empty target array")

        

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
    ccId = cc[cc.id == uniqIds[8]]

    print(ccId["LSSR.EVI.CDI"].isna().any())
    
    # Call class
    inst = ImpactRecoveryMetrics(data = ccId, target = "LSSR.EVI.CDI",eDate = "2011-06-04")
    inst.fitPolyFunc()
    met = inst.getScore()
    print(met)
    fig = inst.visFunc()
    fig.show()

