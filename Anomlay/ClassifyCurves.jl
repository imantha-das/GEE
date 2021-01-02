using DataFrames
using Statistics:mean

#= Classify curvess -------------------------------------------------------------
classification label according to the following criteria.

    Curve type 1 : if there is a minima (arrived at precondition state)
    Curve type 2 : mean value negative (negative gradient)
    Curve type 3 : Mean value positive (positive gradient) 
=#

# Function to give class label based on curve shapes ---------------------------
function classifyCurves(df::DataFrame,metrics::Dict)::DataFrame
    metDf = DataFrame(metrics)
    
    if !ismissing(metDf.precon_x[1])
        metDf[:,:c1] = [1]
    elseif mean(df[:,3]) < 0
        metDf[:,:c1] = [2]
    elseif mean(df[:,3]) > 0
        metDf[:, :c1] = [3]
    end

    return metDf
end



