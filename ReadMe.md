# Phenology Metrics

![](images/Phenology_Metrics.png)

## extPhenLite
* To extract phenology metrics

  `from ectPhenLite import ExtPhen`
  
  `data = pd.read_csv("path/data.csv)`
  
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
