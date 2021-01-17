### ComputeAnomaly.jl
* **Function : computeAnomaly**
  * Calculates the anomaly by computing cumulative difference between the post eruption time series signal and mean pre eruption time series signal.
  * Input arguments
    * ptId : Point Id
        * either give name of point 205936798913767...
        * or set to NaN
    * Data : Dataframe containing columns id, time and target column
    * target : NDVI data
        * i.e LSSR.NDVI
    * eDate : Eruption date
        * i.e : Date(2010,10,26)
    * noPoints : optional argument
        * set ptId to NaN if using this argument
        * give the number of unique points that requires to be computed (more than one geographical point can be calculated)

* **Function : plotAnom**
  * Plots Post eruption time series signal, diff and cumulative diff
  * Input arguments
    * df : dataframe with columns time, post-eruption-target, diff, cumulative diff

### ImapctRecoveryMetrics.jl 
* Julia module to compute impact recovery metrics

  `include(path/ImpactRecoveryMet)`
  
  `df,met = anomMet(data,"113317200868021232614364179530769786479","LSSR.EVI.CDI")`
  
  `anomPlot(df,met)`
  
* anomMet - fits polynomial function to anomaly data and returns anomaly metrics
  * Inputs
    * data - csv file with following columns
      * id:String - id for geographical points
      * time:String - date time 
      * CDI:Float64 - Post eruption anomaly signal
    * idNo:Any - Either string or NaN, if NaN set optional parameter uniqIdIdx to the index required
    * timeThresh:Int64 - time threshold to ignore delay metrics after, By default set to 12 months
  * Outputs
    * df::DataFrame
    * metrics::Dictionary
      * delay - delay of impact effect if present (maxima found only within a range of 12 months default)
      * precondition state - first minima observed, roots of first dervavitive, f'(x) where second dervative, f''(x) > 0  
      * budget - roots of polynomial function f(x)
      * Improvement - first maxima, roots of first dervative f'(x) where second deravative, f''(x) < 0
    
* anomPlot - returns Plots figure with fitted function and metrics
  * Inputs
    * df::DataFrame - Dataframe returned from anomMet function
    * met::Dictionary - Metrics returned from the anomMet function

## ClassifyCurves
