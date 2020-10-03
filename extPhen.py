import os
import numpy as np
import pandas as pd
from peakdetect import peakdetect
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy import integrate

from datetime import timedelta

class ExtPhen():
    def __init__(self,data,target,ID,eDate):
    #def __init__(self,idxTime,serFit,id):
        try:
            self.idxTime = data.index
        except:
            print('unable to find time values in index')
        #self.serFit = data[target]
        self.serFit = data[target].values

        self.eDate = eDate
        self.ID = ID

        #Check if serFit contains all Nan values
        self.isEmpty = np.isnan(self.serFit).all()
        if np.unique(self.serFit).size == 1:
            self.isEmpty = True

        self.mxDtsVal = np.nan
        self.mnDtsVal = np.nan

        self.GreenUpArr = np.nan
        self.BrownDownArr = np.nan
        self.PercUppLeftArr = np.nan
        self.PercUppRightArr = np.nan
        self.midSeasDtVal = np.nan
        self.midSeasDt = np.nan
        self.growSeasArea = np.array([])

        self.erupPhase = None

        self.dfPhen = pd.DataFrame(columns = ['amp','greenupT','greenupV','browndownT','browndownV','LOS','maturityT','maturityV','senescenceT','senescenceV','maxT','maxV','midseasonT','growSeasArea'])

        #Convert series time to datetime object
        self.serTime = self.idxTime.to_series()
        self.serTime = pd.to_datetime(self.serTime,format = '%d/%m/%Y')




# Calculate Metrics
#____________________________________________________________________________________

    def getPhenMet(self,lookRng = 150, deltaVal = 0, isFreq = True, Lperc = 0.1, Uperc = 0.9):
        '''Calculates the peaks (maxima + minima) and find the seasonality metrics'''

        if self.isEmpty == False:
#Calculate maxima & minima ____________________________________________________________________

            #Obtain the frequency of
            freq = self.serTime.iloc[1] - self.serTime.iloc[0]

            #obtain peaks from peak detection function
            if isFreq == True:
                lookRng = int(lookRng / (freq.days))
                peaks = peakdetect(y_axis = self.serFit ,lookahead = lookRng, delta = deltaVal)
            else:
                peaks = peakdetect(y_axis = self.serFit, lookahead = lookRng, delta = deltaVal)

            mxPkIdx,mxPkVal = np.asarray(peaks[0])[:,0], np.asarray(peaks[0])[:,1]
            mnPkIdx,mnPkVal = np.asarray(peaks[1])[:,0],np.asarray(peaks[1])[:,1]

            #Obtain the Maxima / Minima occuring dates using indices
            mxPkDts = np.asarray(self.serTime.iloc[mxPkIdx],dtype = object)
            mnPkDts = np.asarray(self.serTime.iloc[mnPkIdx],dtype = object)

            #Put corresponding max/min dates and values in tuples
            self.mxDtsVal = np.array([(dt,val) for dt,val in zip(mxPkDts,mxPkVal)])
            self.mnDtsVal = np.array([(dt,val) for dt,val in zip(mnPkDts,mnPkVal)])

            #If there is maximum point before/after the first/last  minimum point respectively, remove them.

            if self.mxDtsVal[0,0] < self.mnDtsVal[0,0]:
                self.mxDtsVal = self.mxDtsVal[1:]
                mxPkVal = mxPkVal[1:]
                mxPkDts = mxPkDts[1:]

            if self.mxDtsVal[-1,0] > self.mnDtsVal[-1,0]:
                self.mxDtsVal = self.mxDtsVal[:-1]
                mxPkVal = mxPkVal[:-1]
                mxPkDts = mxPkDts[:-1]

            #Try except block checking if length of mxDtsVal = mnDtsVal + 1, if it isnt the amplitude calcs will be incorrect

            try:
                len(self.mxDtsVal) == len(self.mnDtsVal) - 1
            except:
                print('The maxima and minima doesnot have required lengths to compute phenology params \n mxDtsVal/mnDtsVal length : {} {}'.format(len(self.mxDtsVal),len(self.mnDtsVal)))


#Calculate Phenological Metrics ________________________________________________________________


# Amplitude ------------------------------------------------------------------------------------

            #Calculate Amplitude : ((max - min-left) , (max - min-right))
            Amplitude = list(zip(mxPkVal - mnPkVal[:-1],mxPkVal - mnPkVal[1:]))
            Amplitude = np.asarray(Amplitude)

# green + brown + L Upp Pt + R Upp Pt ---------------------------------------------------------

            #Calculate greenup,browndown,upper left and right points. i.e greenUp = 0.9*amp
            perc_amp_greenup = np.array(mxPkVal - (Amplitude[:,0]*(1-Lperc)))
            perc_amp_browndown = np.array(mxPkVal - (Amplitude[:,1]*(1-Lperc)))
            perc_amp_uppLeft = np.array(mxPkVal - (Amplitude[:,0]*(1-Uperc)))
            perc_amp_uppRight =np.array(mxPkVal - (Amplitude[:,1]*(1-Uperc)))

            #order them in an array : idea is to within a specified date range to identify each of the values for, greenup,browndown, left/right upper percent
            searchValsArr = np.array([(gu,bd,ul,ur) for gu,bd,ul,ur in zip(perc_amp_greenup,  perc_amp_browndown,perc_amp_uppLeft,perc_amp_uppRight)])

            #entire dataset in an ndarray (date,value)
            DtsValArr = np.array([(dt,val) for dt,val in zip(self.serTime,self.serFit)])

            #Maxima - Minima Date Ranges
            searchDtsRangeArr = np.array(list(zip(mxPkDts,mnPkDts[:-1],mxPkDts,mnPkDts[1:])))

            #Nested array containing 8 values within a subarray:
            # [[mxDt,mnDt_left,mxDt,mnDt_right,gup_val, bdwn_val,upp_left,upp_right],[...]]
            searchDtsRangeAmp = np.hstack((searchDtsRangeArr,searchValsArr))

            GreenUp,PercUppLeft = [],[]
            for i in searchDtsRangeAmp:
                # within the first two date values in SchDtRanAmp, store in temp array
                dateidx = np.where(np.logical_and(DtsValArr[:,0] >= i[1], DtsValArr[:,0] <= i[0]))
                temp_datevalArr = DtsValArr[dateidx]

                #Search for green up
                closestValIdx1 = (np.abs(temp_datevalArr[:,1]-i[4])).argmin()
                GreenUp.append(temp_datevalArr[closestValIdx1])

                #search for percentage upper left
                closestValIdx2 = (np.abs(temp_datevalArr[:,1]-i[6])).argmin()
                PercUppLeft.append(temp_datevalArr[closestValIdx2])

            BrownDown,PercUppRight = [],[]
            for i in searchDtsRangeAmp:
                # Within the second two date values items 3 & 4, store in tempdate val array
                dateidx = np.where(np.logical_and(DtsValArr[:,0] >= i[2], DtsValArr[:,0] <= i[3]))
                temp_datevalArr = DtsValArr[dateidx]

                #Search browndown
                closestValIdx1 = (np.abs(temp_datevalArr[:,1]-i[5])).argmin()
                BrownDown.append(temp_datevalArr[closestValIdx1])

                #search for percentage upper right
                closestValIdx2 = (np.abs(temp_datevalArr[:,1]-i[7])).argmin()
                PercUppRight.append(temp_datevalArr[closestValIdx2])

            self.GreenUpArr = np.asarray(GreenUp)
            self.BrownDownArr = np.asarray(BrownDown)
            self.PercUppLeftArr = np.asarray(PercUppLeft)
            self.PercUppRightArr = np.asarray(PercUppRight)


# Mid Season Metrics ---------------------------------------------------------------------------

            self.midSeasDt = self.PercUppLeftArr[:,0] + ((self.PercUppRightArr[:,0] - self.PercUppLeftArr[:,0]) / 2)

# Calculate growing season Area ----------------------------------------------------------------

            for gu,bd in zip(self.GreenUpArr[:,0],self.BrownDownArr[:,0]):

                x_TS = DtsValArr[(DtsValArr[:,0] >= gu) & (DtsValArr[:,0] <= bd)][:,0]
                y_val = DtsValArr[(DtsValArr[:,0] >= gu) & (DtsValArr[:,0] <= bd)][:,1]

                #Convert x_TS values from timestamps to days i.e 0,8,16,24,... since you cannot                     intergrate over timestamps
                x_days = np.arange(start = 0, stop = freq.days * len(x_TS), step = freq.days)

                #Calculate the area below the greenup and browndown date. Model the area as a                       rectangle and use average greenup value and browndown value to obtain heigh
                height = (y_val[0]+y_val[-1])/2
                width = x_days[-1]
                areaBelow = height * width

                # total area under datapoints
                fullArea = integrate.simps(y_val,x_days)

                #area between greenup and browndown
                self.growSeasArea = np.append(self.growSeasArea, fullArea-areaBelow)

# Erup Phase -----------------------------------------------------------------------------------

            greenupDt = np.array([(i[0],'green up') for i in self.GreenUpArr])
            maturityDt = np.array([(i[0], 'maturity') for i in self.PercUppLeftArr])
            senesenceDt = np.array([(i[0], 'senescence') for i in self.PercUppRightArr])
            browndownDt = np.array([(i[0], 'brown down') for i in self.BrownDownArr])

            gmsbLs = []
            for g,m,s,b in zip(greenupDt,maturityDt,senesenceDt,browndownDt):
                gmsbLs.append(g)
                gmsbLs.append(m)
                gmsbLs.append(s)
                gmsbLs.append(b)

            gmsbArr = np.asarray(gmsbLs)

            for idx,val in enumerate(gmsbArr):
                try:
                    if self.eDate >= val[0] and self.eDate < gmsbArr[idx+1][0]:
                        self.erupPhase = val[1]
                except IndexError:
                    self.erupPhase = np.nan
# Build up dataframe ---------------------------------------------------------------------------

            ampArr = np.around((Amplitude[:,0]+Amplitude[:,1])/2, decimals = 3)
            self.dfPhen['amp'] = ampArr[:]
            self.dfPhen['greenupT'] = self.GreenUpArr[:,0]
            self.dfPhen['greenupV'] = self.GreenUpArr[:,1].astype('float')
            self.dfPhen['browndownT'] = self.BrownDownArr[:,0]
            self.dfPhen['browndownV'] = self.BrownDownArr[:,1].astype('float')
            self.dfPhen['LOS'] = (self.dfPhen['browndownT'] - self.dfPhen['greenupT']).dt.days
            self.dfPhen['maturityT'] = self.PercUppLeftArr[:,0]
            self.dfPhen['maturityV'] = self.PercUppLeftArr[:,1].astype('float')
            self.dfPhen['senescenceT'] = self.PercUppRightArr[:,0]
            self.dfPhen['senescenceV'] = self.PercUppRightArr[:,1].astype('float')
            self.dfPhen['maxT'] = self.mxDtsVal[:,0]
            self.dfPhen['maxV'] = self.mxDtsVal[:,1].astype('float')
            if freq == timedelta(days = 1):
                self.dfPhen['midseasonT'] = self.midSeasDtVal[:,0]
            else:
                self.dfPhen['midseasonT'] = self.midSeasDt
            self.dfPhen['growSeasArea'] = self.growSeasArea

            # Changed by Seb 2020-04-14 10:22:04
            self.dfPhen['time'] = self.dfPhen[['greenupT','browndownT','maturityT','senescenceT','maxT','midseasonT']].max(axis = 1).dt.year
            self.dfPhen['time'] = pd.to_datetime(self.dfPhen['time'] , format='%Y')
            self.dfPhen['id'] = self.ID
            self.dfPhen = self.dfPhen.set_index(['id','time'])


        else:
            self.dfPhen['amp'] = np.array([np.nan])
            self.dfPhen['greenupT'] = np.array([np.datetime64('nat')])
            self.dfPhen['greenupV'] = np.array([np.nan])
            self.dfPhen['browndownT'] = np.array([np.datetime64('nat')])
            self.dfPhen['browndownV'] = np.array([np.nan])
            self.dfPhen['LOS'] = np.array([np.nan])
            self.dfPhen['maturityT'] = np.array([np.datetime64('nat')])
            self.dfPhen['maturityV'] = np.array([np.nan])
            self.dfPhen['senescenceT'] = np.array([np.datetime64('nat')])
            self.dfPhen['senescenceV'] = np.array([np.nan])
            self.dfPhen['maxT'] = np.array([np.datetime64('nat')])
            self.dfPhen['maxV'] = np.array([np.nan])
            self.dfPhen['time'] = np.array([self.serTime[0]])
            self.dfPhen['id'] = self.ID
            self.dfPhen = self.dfPhen.set_index(['id','time'])

        #return dataframe
        return self.dfPhen

# Visualization
#____________________________________________________________________________________

    def visPlot (self):

        '''Method that plots Time Series and Phenological metrics'''
        if self.isEmpty == False:
            #create two subplots
            Fig = make_subplots(rows = 2, cols = 1,subplot_titles = ['Timeseries','Phenology Mertics'])
            Fig.update_layout(width = 800,height = 600,template = 'plotly_white')

# subplot 1 : Peaks ---------------------------------------------------------------------------

            # Trace to add fitted line (spline) to timeseries subplot
            Fig.add_trace(go.Scatter(x = self.serTime,y = self.serFit,mode = 'lines',line = dict(color = 'steelblue',width = 2), name = 'Fitted Spline'),row = 1, col = 1)

            #Trace to add Maxima to fig
            Fig.add_trace(go.Scatter(x = self.mxDtsVal[:,0],y = self.mxDtsVal[:,1],mode = 'markers',marker = dict(color = 'red',size = 5,symbol = 'cross'),name = 'Maxima'),row = 1, col = 1)

            #Trace to add Minima to fig
            Fig.add_trace(go.Scatter(x = self.mnDtsVal[:,0],y = self.mnDtsVal[:,1], mode = 'markers',marker = dict(color = 'peru',size = 5,symbol = 'cross'),name = 'Minima'),row = 1, col = 1)

            #Add axis titles
            Fig.update_xaxes(title_text ='Time',row = 1,col = 1)
            Fig.update_yaxes(title_text ='ndvi',row = 1, col = 1)


# Subplot 2 : Add phenology metrics ------------------------------------------------------------

            # Trace to add fitted line (spline) to phenology subplot
            Fig.add_trace(go.Scatter(x = self.serTime,y = self.serFit,mode = 'lines',line = dict(color = 'steelblue',width = 2),name = 'Fitted Spline'),row = 2, col = 1)

            # Trace to add Green up dates to phenology subplot
            Fig.add_trace(go.Scatter(x = self.GreenUpArr[:,0],y = self.GreenUpArr[:,1],mode = 'markers',marker = dict(color = 'olive',size = 5,symbol = 'circle'),name = 'Green Up'),row = 2, col = 1)

            #Trace to add Brown down dates to phenology subplot
            Fig.add_trace(go.Scatter(x = self.BrownDownArr[:,0], y = self.BrownDownArr[:,1],mode = 'markers',marker = dict(color = 'burlywood',size = 5,symbol = 'circle'),name = 'Brown Down'),row = 2, col = 1)

            #Trace to add Maturity dates to Phenology subplot
            Fig.add_trace(go.Scatter(x = self.PercUppLeftArr[:,0], y = self.PercUppLeftArr[:,1], mode = 'markers', marker = dict(color = 'lightseagreen', size = 5, symbol = 'circle'), name = 'Maturity'), row = 2, col = 1)

            #Trace to add Senescence dates to Phenology subplot
            Fig.add_trace(go.Scatter(x = self.PercUppRightArr[:,0], y = self.PercUppRightArr[:,1], mode = 'markers', marker = dict(color = 'darksalmon', size = 5, symbol = 'circle'), name = 'Senescence'), row = 2, col = 1)

            # Update axis titles
            Fig.update_xaxes(title_text ='Time',row = 2,col = 1)
            Fig.update_yaxes(title_text ='ndvi',row = 2, col = 1)

            Fig.add_shape(go.layout.Shape(type = 'line',x0 = self.eDate,y0=np.min(self.mnDtsVal[:,1]),x1 = self.eDate,y1 = np.max(self.mxDtsVal[:,1]),name = 'eruption date', line = dict(color = 'dimgray',dash = 'dash')),row = 2,col = 1)

            return Fig

        else:
            print ('Fitted values are null')
