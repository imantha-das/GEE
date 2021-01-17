# Big Earth Observational Data for Volcanic Impact Assessment
![](images/dash-app.gif)

## Impact Recovery Metrics
![](images/ImpactRecoveryMet_Jl.PNG)
* Finds impact and recovery metrics by fitting a polynomial function to anomaly time series signal.

### ImpactRecoveryMetrics.py
* Python class to compute impact recovery metrics
* Example code

  `from impactRecoveryMetrics import ImpactRecoveryMetrics`
  
  `inst = ImpactRecoveryMetrics(data = ccId, target = "LSSR.EVI.CDI",eDate = "2011-06-04",deg = 4, timeThresh:int = 12)`
  
  `inst.fitPolyFunc()`
  
  `inst.getScore()`
  
  `inst.visFunc()`
* Inputs
  * data : Pandas DataFrame cointaining data for single geographical point
  * target : satelite ndvi dataset 
  * eDate : eruption date
  * deg : degree of polynomial (default = 4)
  * timeThresh : time threshold to capture a delay effect
* Outputs
  * fitpolyFunc() method returns a dictionary with polynomial equation, first and second deravative equations, predicted values
  * getScore() method returns a pandas dataframe with captured metrics
      * delay : delayed impact of vegetation health
      * precon : precondition state (back to mean amplitude of time series ndvi signal)
      * improvDeclPt : improvement decline point. Further impact after recovery
      * budget : neutral point in terms of energy
      * T,TS,V : refer to tome in terms of months, timestamp and ndvi value
  * visFunc() metod returns a plot with metrics and anomaly signal.
  
## Phenology Metrics

![](images/Phenology_Metrics.png)

### extPhenLite.py - python class
* To extract phenology metrics

  `from extPhenLite import ExtPhen`
  
  `data = pd.read_csv("path/data.csv")`
  
  `inst = ExtPhen(data = data, target = "fitted.values", eDate = "2010-10-26")`
  
  `phenMetrics = inst.getPhenMet()`
  
  `erupPhase = inst.erupPhase`
  
  `#To visualize`
  
  `fig = inst.visPlot()`
  
  `fig.show()`

* Inputs
  * data - csv file with containing the following columns
    * time::str
    * fitted.values::float
  * eDate::str - eruption date 
* Outputs
  * inst.getPhenMet::pd.DataFrame - Outputs the phenology metrics captured in the time series
    * amp - amplitude of peaks
    * greenupT/greenupV - season start time/value, attain 10% of amplitude from left side
    * browndownT/browndownV - season end time/ value, attain 10% of amplitude from right side
    * matuarityT/matuarityV - matuarity time/value, attain 90% of ammplitude from left side
    * senescenceT/senescenceV - senescence time/value, attain 90% of amplitude from right side
    * LOS - length of season (browndownT - greenupT)
    * maxT - time at maximum value within a season
    * maxV - maximum value within a season
    * midseasonT - time at mid point of the season
    * growSeasArea - area below greenup and browndown dates
  * inst.erupPhase::str - season during eruption phase 
  * inst.visPlot::plotly.graph.objs.Figure
