import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime,timedelta
import sympy
import sys

class ImpactRecoveryMetrics(object):
    """Fits a polynomial function and computes the impact an recover metrics from the time series signal"""

    def __init__(self,data,target,eDate,deg = 4, maxT_min = 100):
        """ data : Pandas dataframe, must contain columns time (time since eruption) & diff : anomaly
            target : Target column name in string
            eDate : Eruption date in string format YYYYMMDD
            deg : Degree of the polynomial
            maxT_min : Removes any early maxima detected (in days), i.e any maxima detected before 1000 days isnt considered as recovery"""

        self.data= data
        self.target = target
        self.deg = deg
        self.maxT_min = maxT_min

        self.polyEq = np.nan
        self.polyEqDiffFirst = np.nan
        self.polyEqDiffSecond = np.nan
        self.yHat = np.array([])
        self.scoreDict = np.nan

        # Convert time column to date time values
        self.data['time'] = pd.to_datetime(data['time'], format = '%Y-%m-%d')

        # Convert eruption date to datetime value
        try:
            self.eDate = pd.to_datetime(eDate, format = '%Y-%m-%d')
        except:
            print('Eruption date in incorrect format, Requires YYYY-MM-DD')
            sys.exit(1)

        # Compute time since eruption in days
        self.tmSinErup = self.data['time'] - self.eDate
        self.tmSinErup = np.array([i.days for i in self.tmSinErup])

        # Store anomaly values in variable called deltaAnom
        self.deltaAnom = self.data[self.target]

        # Check if deltAnom contains all 0's
        self.isEmpty = False
        self.isEmpty = np.isnan(self.deltaAnom).all()


        # Check if diff column has any nan values, if so replace with mean values
        if np.any(np.isnan(self.deltaAnom)):
            self.deltaAnom = np.where(~np.isnan(self.deltaAnom),self.deltaAnom, np.nanmean(self.deltaAnom))
    # _______________________________________________________________________________________________________

    def fitPolyFunc(self):
        """Method to fit polynomial function to data"""

        # Get polynomial function intercept and coeffcients -----------------------------------------------

        poly = np.polyfit(x = self.tmSinErup, y = self.deltaAnom,deg = self.deg)

        # Construct symbolic polynomial equation ----------------------------------------------------------

        # Define sympy symbol object
        self.z = sympy.symbols('z')

        # Save Polynomial equation intercept
        self.polyEq = poly[self.deg]

        # Save Polynomial equation coeffcients
        for i in range(len(poly)-1):
            self.polyEq += poly[i] * self.z**(self.deg-i)

        # First Deravative
        self.polyEqDiffFirst = sympy.diff(self.polyEq,self.z)
        # Second Deravative
        self.polyEqDiffSecond = sympy.diff(self.polyEqDiffFirst,self.z)

        # Get predicted values from polynomial equation ---------------------------------------------------
        self.yHat = np.array([self.polyEq.subs(self.z,i) for i in self.tmSinErup]).astype('float')

        return self.yHat

    # ______________________________________________________________________________________________________

    def visFunc(self):
        """ Method to visualize polynomial function and Impact-Recovery points """

        if self.yHat.size != 0:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x = self.tmSinErup,
                    y = self.data[self.target],
                    mode = 'markers',
                    marker = dict(color = 'gainsboro', size = 4),
                    name = 'Anomaly'
                )
            )
            fig.add_trace(
                go.Scatter(
                    x = self.tmSinErup,
                    y = self.yHat,
                    mode = 'lines',
                    marker = dict(color = 'lightsteelblue',size = 2),
                    name = 'Fitted Function, deg = '+ str(self.deg)
                )
            )

            fig.update_layout(template = 'plotly_white', height = 600, width = 800, showlegend = True, title = 'Maximum Impact and Recovery of Vegetation')
            fig.update_xaxes(title_text = 'Time since eruption (days)')
            fig.update_yaxes(title_text = 'ndvi')

            if self.scoreDict is not np.nan:

                # Add delayT, to graph
                if self.scoreDict['delayT'].size > 0:
                    fig.add_trace(
                        go.Scatter(
                            x = self.scoreDict['delayT'],
                            y = np.array([0]), mode = 'markers',
                            marker = dict(symbol = 'circle', color = 'darkolivegreen', size = 9),
                            name = 'delay'
                            )
                        )

                # Add neutral to graph
                if self.scoreDict['neutralT'].size > 0:
                    fig.add_trace(
                        go.Scatter(
                        x = self.scoreDict['neutralT'],
                        y = np.array([0]),
                        mode = 'markers',
                        marker = dict(symbol = 'circle', color = '#ffa500',size = 9),
                        name = 'neutral'
                        )
                    )

                # Add minV to graph
                if (self.scoreDict['minT'].size > 0) & (self.scoreDict['minV'].size > 0):
                    fig.add_trace(
                        go.Scatter(
                            x = self.scoreDict['minT'],
                            y = self.scoreDict['minV'],
                            mode = 'markers',
                            marker = dict(symbol = 'circle', color = 'orangered', size = 9),
                            name = 'Max Impact'
                        )
                    )

                if (self.scoreDict['maxT'].size > 0) & (self.scoreDict['maxV'].size > 0):
                    fig.add_trace(
                        go.Scatter(
                            x = self.scoreDict['maxT'],
                            y = self.scoreDict['maxV'],
                            mode = 'markers',
                            marker = dict(symbol = 'circle', color = 'limegreen', size = 9),
                            name = 'Recovery'
                            )
                        )

        else:
            print('Run fitPolyFunc method first ')

        return fig

    # ______________________________________________________________________________________________________

    def getScore(self):
        """ Compute Impact and Recovery values and corresponding time by identifying the extrema points.
         First deravative, dy/dx < 0 (or -ve) indicates the curve going down, corresponding to a maxima.
         First deravative, dy/dx > 0 (or +ve) indicates the curve going up correspondin to a minima """

        if self.isEmpty == False:

            # Time range of signal
            tmRng = (np.float(self.tmSinErup[0]), np.float(self.tmSinErup[-1]))

            #Extrema points
            # ---------------------------------------------------------------------------------------------------

            # Compute all extrema points
            allMaxMinRts = np.array(sympy.solveset(self.polyEqDiffFirst, self.z, domain = sympy.S.Reals).args).astype('float')
            # Compute extrema points within the time interval
            MaxMinRtsInRng = allMaxMinRts[(allMaxMinRts >= tmRng[0]) & (allMaxMinRts <= tmRng[1])]

            # Second deravative check to identify Maxima from Minima
            secondDerChk = np.array([self.polyEqDiffSecond.subs(self.z,i) for i in MaxMinRtsInRng])
            maxRts = MaxMinRtsInRng[secondDerChk < 0]
            minRts = MaxMinRtsInRng[secondDerChk > 0]

            # Find the corresponding y vals at maxRts and minRts
            maxRtVals = np.array([self.polyEq.subs(self.z,i) for i in maxRts]).astype('float')
            minRtVals = np.array([self.polyEq.subs(self.z,i) for i in minRts]).astype('float')

            # Roots at y = 0
            # ----------------------------------------------------------------------------------------------------

            allRts0 = np.array(sympy.solveset(self.polyEq,self.z,domain = sympy.S.Reals).args).astype('float')
            rts0 = allRts0[(allRts0 >= tmRng[0]) & (allRts0 <= tmRng[1])]

            # Root dictionary
            # ---------------------------------------------------------------------------------------------------

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
            # -------------------------------------------------------------------------------------------------

            #if there is any roots @ y= 0 -> Then check for the first deravative if negative - Its potential   candidate for delayT
            if (rts0.size > 0):
                delayDerChk = np.array([self.polyEqDiffFirst.subs(self.z,i) for i in rts0]).astype('float')
                delayT = rts0[delayDerChk < 0]
                # if there is a minima, select all roots that have x values less than the x value of minima
                if (delayT.size > 0) & (minRts.size > 0):
                    # Select delayT values occuring before minima
                    delayT = delayT[delayT < minRts[0]]
                    #if there are still more than one root pick first
                    if delayT.size > 0:
                        delayT = np.array([delayT[0]])
                    # After inner filter if there isnt any values -> Return empty array
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
            # ------------------------------------------------------------------------------------------------

            #if there is any roots @ y = 0 -> Then check for the first dervative if it is positive : Its potential candiate for neutralT
            if (rts0.size > 0):
                neutDerChk = np.array([self.polyEqDiffFirst.subs(self.z,i) for i in rts0]).astype('float')
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
            # --------------------------------------------------------------------------------------------------

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
                if (self.yHat[-1] > 0):
                    maxT = np.array([tmRng[1]])
                    maxV = np.array([self.yHat[-1]])
                else:
                    maxT = np.array([])
                    maxV = np.array([])

            # minT, minV
            # -----------------------------------------------------------------------------------------------

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
                    if (self.yHat[-1] < 0):
                        minT = np.array([tmRng[1]])
                        minV = np.array([self.yHat[-1]])
                    else:
                        minT = np.array([])
                        minV = np.array([])

            # Remove delay or neutral if they occure after maxT : minT is already accounted for -- check line 379
            # ------------------------------------------------------------------------------------------------------

            if (maxT.size > 0) & (neutralT.size) > 0:
                if neutralT[0] > maxT[0]:
                    neutralT = np.array([])

            if maxT.size > 0 & delayT.size > 0:
                if delayT[0] > maxT[0]:
                    delayT = np.array([])


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

            # Score Dictionary
            # ---------------------------------------------------------------------------------------------------

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

            # Replace empty arrays with nan,nat and 0's
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

            return dfScore


if __name__ == '__main__':

    data = pd.read_csv('EE/GEE lite/Imantha_anomaly3.csv')

    inst = ImpactRecoveryMetrics(data = data, target = 'diff',eDate = '2010-10-26')

    inst.fitPolyFunc()
    score= inst.getScore()
    print(score)

    # To visualize
    #fig = inst.visFunc()
    #fig.show()

