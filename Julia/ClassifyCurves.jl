using DataFrames
using Statistics:mean

#= Classify curvess -------------------------------------------------------------
classification label according to the following criteria.

    Curve type 1 : if there is a minima (arrived at precondition state)
    Curve type 2 : mean value negative (negative gradient)
    Curve type 3 : Mean value positive (positive gradient) 
    
=#

function classifyCurves(metrics::Dict)::DataFrame
    
    if !ismissing(metrics["preconT"][1])
        metrics["c1"] = [1]
    elseif mean(metrics["fit"]) < 0
        metrics["c1"] = [2]
    elseif mean(metrics["target"]) > 0
        metrics["c1"] = [3]
    end

    selectedKeys = Array{String}(["id","delayT","delayTS","delayV","preconT","preconTS","preconV","improvDeclT","improvDeclTS","improvDeclV","budgetT","budgetTS","budgetV","c1"])

    metSelected = Dict{String,Any}()
    for k in selectedKeys
            push!(metSelected,k => metrics[k][1])
    end
    df = DataFrame(metSelected)
    select!(df,[:id,:delayT,:delayTS,:delayV,:preconT,:preconTS,:preconV,:improvDeclT,:improvDeclTS,:improvDeclV,:budgetT,:budgetTS,:budgetV,:c1])

    return df
end

# ____________________________________________________________
# Function Call 

#=
using SQLite
include("ImpactRecoveryMet.jl")

db = SQLite.DB("D:/GEE_Project/Databases/database.db")
anomalyM = DataFrame(SQLite.DBInterface.execute(db,"SELECT * FROM anomalyM"))

met = anomMet(anomalyM,NaN,"LSSR.EVI.CDI",uniqIdIdx =22)
p = anomPlot(met)
@show classifyCurves(met)

# Check classes for point ID 22,23 
=#