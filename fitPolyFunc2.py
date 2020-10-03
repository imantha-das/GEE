import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score

from plotly import graph_objects as go
from plotly.subplots import make_subplots

from datetime import timedelta,datetime

import sympy

class FitPolyFunc2 ():
    '''Class to Fit a polynomial function to Anomaly'''

    def __init__(self,data,target,eDate, ID, deg = 4,testSize = 0.25, rs=42,maxT_min = 100):

        self.data = data
        self.target = target
        self.eDate = eDate
        self.ID = ID
        self.deg = deg
        self.testSize = testSize
        self.rs = rs
        self.maxT_min = maxT_min

        if 'interval' in self.data.columns:  # If postI
            self.serTime = self.data.index
        elif 'month' in self.data.columns:   # If postM
            self.serTime = self.data.index+pd.Timedelta('15 days')
        else:                           # If postY
            self.serTime = self.data.index+pd.Timedelta('183 days')

        self.serTime = self.serTime.to_numpy()

        self.deltaAnom = self.data[target].to_numpy()
        self.tmSinErup = (self.data.index-self.eDate).days.to_numpy()

        #Check if deltaAnom contain all 0's
        self.isEmpty = False
        self.addMetricsToPlot = False

        self.isEmpty = np.isnan(self.deltaAnom).all()

        if np.unique(self.deltaAnom).size == 1:
            self.isEmpty = True


        #Replace missing values with mean values
        if np.any(np.isnan(self.deltaAnom)):
            self.deltaAnom = np.where(np.isnan(self.deltaAnom),np.ma.array(self.deltaAnom, mask = np.isnan(self.deltaAnom)).mean(axis = 0),self.deltaAnom)

        if np.any(np.isnan(self.tmSinErup)):
            self.tmSinErup = np.where(np.isnan(self.tmSinErup),np.ma.array(self.tmSinErup, mask = np.isnan(self.tmSinErup)).mean(axis = 0),self.tmSinErup)

        #Test Statistics
        self.testStats = {'MAE' : np.nan, 'MSE' : np.nan, 'R2' : np.nan}

        #predData atrributes
        self.yPred = np.nan

        #getPolyEq attributes
        self.z = np.nan
        self.polyEq = np.nan
        self.polyDiffFirst = np.nan
        self.PolyDiffSecond = np.nan

        #calcRoots
        self.rootDict = np.nan
        self.scoreDict = np.nan


# fitPredData
# ________________________________________________________________________________________________

    def fitPredData(self):
        '''Fits model function to predict anomaly values (Response variable). Explanatory variable used for prediction is time since eruption in days. Return the predicted anaomaly values in an ndarray '''

#Fit Data ----------------------------------------------------------------------------------------
        if self.isEmpty == False:
            x_train,x_test,y_train,y_test = train_test_split(self.tmSinErup,self.deltaAnom, test_size = self.testSize, random_state = self.rs)
            # breakpoint()
            xTrainArr = x_train.reshape(-1,1)
            yTrainArr = y_train.reshape(-1,1)
            xTestArr = x_test.reshape(-1,1)
            yTestArr = y_test.reshape(-1,1)

            #poly and clf objects
            poly = PolynomialFeatures(degree = self.deg)
            clf = linear_model.LinearRegression()

            #Transform x train values / train classifier based on transformed values
            xTrainTrans = poly.fit_transform(xTrainArr)
            clf.fit(xTrainTrans,yTrainArr)


#Test Statistics ---------------------------------------------------------------------------------

            #Transform x test values / predict y values based on the transformed values
            xTestTrans = poly.transform(xTestArr)
            yTestPred = clf.predict(xTestTrans)

            #Calculate metrics to measure the performance of classifier
            mean_abs_error = np.mean(np.absolute(yTestPred - yTestArr))
            mean_sqd_error = np.mean((yTestPred - yTestArr)**2)
            r_squared = r2_score(yTestPred, yTestArr)

            self.testStats['MAE'] = mean_abs_error
            self.testStats['MSE'] = mean_sqd_error
            self.testStats['R2'] = r_squared

# Predicted data using transformed values --------------------------------------------------------

            #Reshape all x data / transform all x data / obtain a prediction for y for all x data
            tmSinErupRshp = self.tmSinErup.reshape(-1,1)
            xTrans = poly.transform(tmSinErupRshp)
            self.yPred = clf.predict(xTrans)
            self.yPred = self.yPred.flatten()

# Polynomial equation and lamdified equations ----------------------------------------------------

            #Use sympy to form symbolic equation, which also can predict yPred using equation
            self.z = sympy.symbols('z')
            self.polyEq = clf.intercept_
            for idx,val in enumerate(clf.coef_[0]):
                self.polyEq = self.polyEq + val * self.z**idx

            #First Deravative
            self.polyDiffFirst = sympy.diff(self.polyEq,self.z)

            # Second deravative
            self.polyDiffSecond = sympy.diff(self.polyDiffFirst,self.z)

            return self.yPred


#VisFunc Method
# _______________________________________________________________________________________________

    def visFunc(self):
        if self.isEmpty == False:
            fig = make_subplots(rows = 2, cols = 1,subplot_titles = ['Anomaly', 'Delta + Fitted Function'])

            #split target at second datapoint and ass '.anomaly'
            ndviVals = ".".join(self.target.split(".", 2)[:2])+'.anomaly'

            fig.add_trace(go.Scatter(x = self.data.index, y = self.data[ndviVals], mode = 'lines', line = dict(
                color ='steelblue', width = 2), name = 'anomaly'), row = 1, col = 1)

            fig.add_trace(go.Scatter(x = self.tmSinErup, y = self.deltaAnom, mode = 'markers', marker = dict(color ='gainsboro', size = 2),name = 'delta'), row = 2, col = 1)

            fig.add_trace(go.Scatter(x = self.tmSinErup, y = self.yPred, mode = 'lines',line = dict(color = 'lightsteelblue', width = 2), name = 'fitted function'),row = 2, col = 1)

            fig.update_layout(template = 'plotly_white',height = 600, width = 800, showlegend = True)
            fig.update_xaxes(title_text = 'Time',row =1 , col =1 )
            fig.update_yaxes(title_text = 'ndvi',row = 1, col = 1)
            fig.update_xaxes(title_text = 'Time since eruption (days)',row =2 , col =1 )
            fig.update_yaxes(title_text = 'ndvi',row = 2, col = 1)

#add scores
# ------------------------------------------------------------------------------------------------------------------

            #if self.scoreDict['delayT'].size > 0:
                #fig.add_shape(go.layout.Shape(type = 'line',x0 = self.scoreDict['delayT'][0],y0 = np.min(self.yPred), x1 = self.scoreDict['delayT'][0], y1 = np.max(self.yPred), line = dict(dash = 'dash', width = 3, color = 'darkolivegreen'), name = 'delay'),row = 2,col = 1)

            #if self.scoreDict['neutralT'].size > 0:
                #fig.add_shape(go.layout.Shape(type = 'line',x0 = self.scoreDict['neutralT'][0],y0 = np.min(self.yPred), x1 = self.scoreDict['neutralT'][0], y1 = np.max(self.yPred), line = dict(dash = 'dash', width = 3, color = '#ffa500'), name = 'neutral'),row = 2,col = 1)

            if self.scoreDict['delayT'].size > 0:
                fig.add_trace(go.Scatter(x = self.scoreDict['delayT'], y = np.array([0]), mode = 'markers', marker = dict(symbol = 'circle', color = 'darkolivegreen', size = 9), name = 'delay'), row = 2, col = 1)

            if self.scoreDict['neutralT'].size > 0:
                fig.add_trace(go.Scatter(x = self.scoreDict['neutralT'], y = np.array([0]), mode = 'markers', marker = dict(symbol = 'circle', color = '#ffa500',size = 9), name = 'neutral'), row = 2, col = 1)

            if (self.scoreDict['minT'].size > 0) & (self.scoreDict['minV'].size > 0):
                fig.add_trace(go.Scatter(x = self.scoreDict['minT'], y = self.scoreDict['minV'], mode = 'markers', marker = dict(symbol = 'circle', color = 'orangered', size = 9), name = 'min'), row = 2, col = 1)


            if (self.scoreDict['maxT'].size > 0) & (self.scoreDict['maxV'].size > 0):
                fig.add_trace(go.Scatter(x = self.scoreDict['maxT'], y = self.scoreDict['maxV'], mode = 'markers', marker = dict(symbol = 'circle', color = 'limegreen', size = 9), name = 'max'), row = 2, col = 1)
        #if self.addMetricsToPlot == True:
            return fig

        else:
            print('delta Anom contain either 0, nan or constant value for all data')


# get score
# ________________________________________________________________________________________________

    def getScore(self):

        if self.isEmpty == False:

            tmRng = (np.float(self.tmSinErup[0]), np.float(self.tmSinErup[-1]))

# Dervative info,
# first deravative, dy/dx < 0 (or -ve) indicates the curve going down.
# first deravative, dy/dx > 0 (or +ve) indicates the curve going up.

# stationary points
# -----------------------------------------------------------------------------------------------------
            # Find all real roots for all stationary points
            allMaxMinRts = np.array(sympy.solveset(self.polyDiffFirst[0], self.z, domain = sympy.S.Reals).args).astype('float')

            # Max Min roots within Range
            MaxMinRtsInRng = allMaxMinRts[(allMaxMinRts >= tmRng[0]) & (allMaxMinRts <= tmRng[1])]

            # Second deravative check to identify Maxima from Minima
            secondDerChk = np.array([self.polyDiffSecond[0].subs(self.z,i) for i in MaxMinRtsInRng])
            maxRts = MaxMinRtsInRng[secondDerChk < 0]
            minRts = MaxMinRtsInRng[secondDerChk > 0]

            # Find the corresponding y vals at maxRts and minRts
            maxRtVals = np.array([self.polyEq[0].subs(self.z,i) for i in maxRts]).astype('float')
            minRtVals = np.array([self.polyEq[0].subs(self.z,i) for i in minRts]).astype('float')


# Roots at y = 0
# ----------------------------------------------------------------------------------------------------
            allRts0 = np.array(sympy.solveset(self.polyEq[0],self.z,domain = sympy.S.Reals).args).astype('float')
            rts0 = allRts0[(allRts0 >= tmRng[0]) & (allRts0 <= tmRng[1])]

# Root dictionary
# ----------------------------------------------------------------------------------------------------
            self.rootDict = {
                'allMaxMinRts' : allMaxMinRts,
                'MaxMinRtsInRng' : MaxMinRtsInRng,
                'minRts' : minRts,
                'maxRts' : maxRts,
                'minRtVals' : minRtVals,
                'maxRtsVals' : maxRtVals,
                'allRts0' : allRts0,
                'rts0' : rts0
                }

# delayT
# -----------------------------------------------------------------------------------------------------------------
            #if there is any roots @ y= 0 -> Then check for the first deravative if negative - Its potential   candidate for delayT
            if (rts0.size > 0):
                delayDerChk = np.array([self.polyDiffFirst[0].subs(self.z,i) for i in rts0]).astype('float')
                delayT = rts0[delayDerChk < 0]
                # if there is a minima, select all roots that have x values less than the x value of minima
                if (delayT.size > 0) & (minRts.size > 0):
                    # Select delayT values occuring before minima
                    delayT = delayT[delayT < minRts[0]]
                    #if there are still more than one root pick first
                    if delayT.size > 0:
                        delayT = np.array([delayT[0]])
                    # After inner filter if there isnt any values -> Return empty arrat
                    else:
                        delayT = np.array([])
                # if there no minima, then pick the first value as delayT
                elif (delayT.size > 0):
                    delayT = np.array([delayT[0]])
                #after first filter if there is no delayT value -> Return empty array
                else:
                    delayT = np.array([])
            else:
                delayT = np.array([])


# neutralT
# ------------------------------------------------------------------------------------------------------------------
            #if there is any roots @ y = 0 -> Then check for the first dervative if it is positive : Its potential candiate for neutralT
            if (rts0.size > 0):
                neutDerChk = np.array([self.polyDiffFirst[0].subs(self.z,i) for i in rts0]).astype('float')
                neutralT =  rts0[neutDerChk > 0]
                # if there is a minima, select all roots that have x values greater than the x value of minima
                if (neutralT.size > 0) & (minRts.size > 0):
                    neutralT = neutralT[neutralT > minRts[0]]
                    #If there more than one option to pick.
                    if neutralT.size > 0:
                        neutralT = np.array([neutralT[0]])
                    else:
                        neutralT = np.array([])

                # If there is no minima -> then pick the first value as neutralT
                elif neutralT.size > 0:
                    neutralT = np.array([neutralT[0]])
                #After filtering in the second line if there is (neutralT[neutDerChk]) no neutralT value return an empty array
                else:
                    neutralT = np.array([])
            else:
                neutralT = np.array([])


# maxT, maxV
# ----------------------------------------------------------------------------------------------------------------

            if (maxRts.size > 0):

                # Condition 1: There is max roots and it is one maxima
                if maxRts.size  == 1:
                    # Condition 1.1 : There is no delay
                    if (delayT.size == 0):

                        # Condition 1.2 : Check if values are below the maxT_min Threshold -> if so filter out maxT based on
                        maxT = maxRts [maxRts > self.maxT_min]
                        maxT_idx = np.where(maxRts == maxT)
                        maxV = maxRtVals[maxT_idx]

                        if maxT.size > 0:
                            maxT = np.array([maxT[0]])
                            maxV = np.array([maxV[0]])
                        else:
                            maxT = np.array([])
                            maxV = np.array([])

                    # Condition 1.3:  There is a delay
                    else:
                    # Condition 1.4 : Check if values are before delay -> If so filter them out.

                        maxT = maxRts[maxRts > delayT[0]]
                        maxT_idx = np.where(maxRts == maxT)
                        maxV = maxRtVals[maxT_idx]

                        if maxT.size > 0:
                        # Condition 1.5 : Check if values are before maxT_min threshold -> if so filter them out
                            tmp_maxT = maxT[maxT > self.maxT_min]
                            maxT_idx = np.where(maxT == tmp_maxT)
                            maxT = tmp_maxT
                            maxV = maxV[maxT_idx]

                            # If there is more than one value
                            if maxT.size > 0:
                                maxT = np.array([maxT[0]])
                                maxV = np.array([maxV[0]])
                            else:
                                maxT = np.array([])
                                maxV = np.array([])

                        else:
                            maxT = np.array([])
                            maxV = np.array([])

                # Condition 2 : There are multiple max roots
                else :
                    # Condition 2.1 : There is no delay
                    if delayT.size == 0:

                    # Condition 2.2 : Check if values are before maxT_min threshold -> If so filter them out.
                        maxT = maxRts[maxRts > self.maxT_min]
                        maxT_idx = np.where(maxRts == maxT)
                        maxV = maxRtVals[maxT_idx]

                        if maxT.size > 0:
                        # Select the first value from the filtered values
                            maxT = np.array([maxT[0]])
                            maxV = np.array([maxV[0]])
                        else:
                            maxT = np.array([])
                            maxV = np.array([])

                    # Condition 2.3 : There is a delay
                    else:
                        #Condition 2.4 : Check if there are any values before delay ->  if so filter them out
                        maxT = maxRts[maxRts > delayT[0]]
                        maxT_idx = np.where(maxRts == maxT)
                        maxV = maxRtVals[maxT_idx]

                        # Condition 2.5 : Check if there are any values before maxT_min Threshold -> If they are filter them out
                        if maxT.size > 0:
                            tmp_maxT = maxT[maxT > self.maxT_min]
                            maxT_idx = np.where(maxT == tmp_maxT)
                            maxT = tmp_maxT
                            maxV = maxV[maxT_idx]

                            # If there is one or more than one value
                            if maxT.size > 0:
                                maxT = np.array([maxT[0]])
                                maxV = np.array([maxV[0]])
                            # If there is no values
                            else:
                                maxT = np.array([])
                                maxV = np.array([])
                        # If there was no values
                        else:
                            maxT = np.array([])
                            maxV = np.array([])

            # Condition 3 : There is no maxT roots
            else:
                # Condition 3.1 : Select the last point of the curve, provided that its a maximum
                if (self.yPred[-1] > 0):
                    maxT = np.array([tmRng[1]])
                    maxV = np.array([self.yPred[-1]])
                else:
                    maxT = np.array([])
                    maxV = np.array([])


#minT, minV
# ------------------------------------------------------------------------------------------------------------------

            if minRts.size > 0:
                #If maxT exists
                if maxT.size > 0:
                    # Remove all minT values occuring after maxT

                    minT = minRts[minRts < maxT[0]]
                    minT_idx = np.where(minRts == minT)
                    minV = minRtVals[minT_idx]
                    if minT.size == 1:
                        minT = minT
                        minV = minV
                    elif minRts.size > 1:
                        minT = np.array([minRts[0]])
                        minV = np.array([minRtVals[0]])
                    else:
                        minT =np.array([])
                        minV = np.array([])

                #If maxT doesnt exist
                else:
                    if minRts.size == 1:
                        minT = minRts
                        minV = minRtVals
                    else :
                        minT = np.array([minRts[0]])
                        minV = np.array([minRtVals[0]])
                    # Else satement for o val not required as minRts.size > 0 ensure there is atleast one minima
            # If there are no roots for minT -> Pick the last point as minT if the corresponding minV value is -ve, otherwise spit out an empty aray
            else:
                if maxT.size > 0:
                    minT = np.array([])
                    minV = np.array([])

                else:
                    if (self.yPred[-1] < 0):
                        minT = np.array([tmRng[1]])
                        minV = np.array([self.yPred[-1]])
                    else:
                        minT = np.array([])
                        minV = np.array([])


# delayTS,maxTS,neutralTS,preTS
# ------------------------------------------------------------------------------------------------------

            if delayT.size > 0:
                delayTS = np.array([(self.eDate + timedelta( days = delayT[0])).date()])
            else:
                delayTS = np.array([])
            if minT.size > 0:
                minTS = np.array([(self.eDate + timedelta( days = minT[0])).date()])
            else:
                minTS = np.array([])

            if neutralT.size > 0:
                neutralTS = np.array([(self.eDate + timedelta( days = neutralT[0])).date()])
            else:
                neutralTS = np.array([])

            if maxT.size > 0:
                maxTS = np.array([(self.eDate + timedelta(days = maxT[0])).date()])
            else:
                maxTS = np.array([])

#Score Dictionary
# ----------------------------------------------------------------------------------------------------

            self.scoreDict = {
                'delayT' : delayT,
                'delayTS' : delayTS,
                'minT' : minT,
                'minTS' : minTS,
                'minV' : minV,
                'neutralT' : neutralT,
                'neutralTS' : neutralTS,
                'maxT' : maxT,
                'maxTS' : maxTS,
                'maxV' : maxV
            }

# Replace empty arrays with nan, nat and 0's
# ---------------------------------------------------------------------------------------------------

            key_nan = ['delayT','minT','neutralT','maxT']
            key_nat = ['delayTS', 'minTS','neutralTS', 'maxTS']
            key_0 = ['minV','maxV']

            for key,val in zip(self.scoreDict.keys(),self.scoreDict.values()):
                if val.size == 0:
                    if key in key_nan:
                        self.scoreDict[key] = np.array([np.nan])
                    elif key in key_nat:
                        self.scoreDict[key] = np.array([np.datetime64('nat')])
                    else:
                        self.scoreDict[key] = np.array([0])

# Convert to dataframe
# --------------------------------------------------------------------------------------------------

            dfScore = pd.DataFrame.from_dict(self.scoreDict)
            dfScore['id'] = self.ID
            dfScore.set_index('id',inplace = True)

            # Convert data types of dfScore
            dfScore[['delayT','minT','neutralT','maxT']] = dfScore[['delayT','minT','neutralT','maxT']].apply(pd.to_timedelta,unit = 'D')

            # Extract day for each of the terms
            dfScore['delayT'] = dfScore['delayT'].dt.days
            dfScore['minT'] = dfScore['minT'].dt.days
            dfScore['neutralT'] = dfScore['neutralT'].dt.days
            dfScore['maxT'] = dfScore['maxT'].dt.days

            # convert delayT, minT, minV,neutralT,maxT,maxV
            dfScore[['delayT','minT','minV','neutralT','maxT','maxV']] = dfScore[['delayT','minT','minV','neutralT','maxT','maxV']].astype('float')

            # Convert TS variables to datetime
            dfScore[['delayTS','minTS','neutralTS','maxTS']] = dfScore[['delayTS','minTS','neutralTS','maxTS']].apply(pd.to_datetime)

            return dfScore

        # Else statement : To return dataframe with nan values

        else:

            self.scoreDict = {
                'delayT' : np.array([np.nan]),
                'delayTS' : np.array([np.datetime64('nat')]),
                'minT' : np.array([np.nan]),
                'minTS' : np.array([np.datetime64('nat')]),
                'minV' : np.array([np.nan]),
                'neutralT' : np.array([np.nan]),
                'neutralTS' : np.array([np.datetime64('nat')]),
                'maxT' : np.array([np.nan]),
                'maxTS' : np.array([np.datetime64('nat')]),
                'maxV' : np.array([np.nan])
            }

            dfScore = pd.DataFrame.from_dict(self.scoreDict)
            dfScore['id'] = self.ID
            dfScore.set_index('id', inplace = True)

            return dfScore

    def get_curveRoots(self):
        if self.isEmpty == False:
            dfRts = pd.DataFrame()

            dfRts['minRts'] = np.array([self.rootDict['minRts'].size])
            dfRts['maxRts'] = np.array([self.rootDict['maxRts'].size])
            dfRts['rts0'] = np.array([self.rootDict['rts0'].size])
            dfRts['id'] = self.ID
            dfRts.set_index('id', inplace = True)

            return dfRts

        else:
            dfRts = pd.DataFrame()

            dfRts['minRts'] = np.array([np.nan])
            dfRts['maxRts'] = np.array([np.nan])
            dfRts['rts0'] = np.array([np.nan])
            dfRts['id'] = self.ID
            dfRts.set_index('id', inplace = True)

            return dfRts
