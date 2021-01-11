module ImpactRecoveryMetrics

using DataFrames
using Query
using Plots
using Polynomials: fit,derivative,roots
using CSV
using Dates


#To use functions explicitly
export anomMet,anomPlot 

# Function to convert months in floats to Dates : Used for polynomial roots
function convDateTime(x)
        fm,m = modf(x)
        Day(Int(round(30.416 * fm))) + Month(Int(m))
end

# Function to compute anomaly metrics
function anomMet(data::DataFrame, idNo::Any, target::String ; uniqIdIdx::Int64 = 1,timeThresh::Int64 = 12)::Dict
        if typeof(idNo) == String 
                ptId = idNo
        else
                uniqIds = unique(data.id)
                ptId = uniqIds[uniqIdIdx]
        end

        # Filter dataframe based on point id
        df = @from row in data begin
                @where row.id == ptId
                @select {id = row.id, time = row.time, target = row[Symbol(target)]}
                @collect DataFrame
        end

        df.time = Date.(df.time, dateformat"Y-m-d H:M:S")
        df.id = String.(df.id)
        # drop missing values
        dropmissing!(df)
        df.target = Float64.(df.target)

        rename!(df, Array{String}(["id","time", target]))

        # Fit polynomial function to data and compute deravatives
        x = collect(1:nrow(df))
        y = df[:,Symbol(target)]
        polyEq = fit(x,y,4)

        ŷ = polyEq.(x)

        polyEq¹ = derivative(polyEq)
        polyEq² = derivative(polyEq¹)

        #add values into dataframe
        df[:,"months_since_impact"] = x
        df[:,"yhat"] = ŷ 

        # Compute roots at y = 0
        zeroPts = roots(polyEq)
        zeroIdx = findall(x -> iszero(x), imag.(zeroPts))
        zeroallRts = real.(zeroPts[zeroIdx]) 
        zeroRts = intersect(zeroallRts[zeroallRts .> 0], zeroallRts[zeroallRts .< x[end]])
        zeroRts = zeroRts[zeroRts .> timeThresh] # zero roots altrady filtered

        # Compute Stationary Points
        maxminPts¹ = roots(polyEq¹)
        maxminIdx¹ = findall(x -> iszero(x),imag.(maxminPts¹)) # get index of real roots
        maxminallRts¹ = real.(maxminPts¹[maxminIdx¹]) # get real component 
        maxminRts¹ = intersect(maxminallRts¹[maxminallRts¹ .> 0], maxminallRts¹[maxminallRts¹ .< x[end]]) # remove rts outside months since impact range
       
        # Find maxima and minima 
        secondDerRts = polyEq².(maxminRts¹)
        minIdx = findall(>(0),secondDerRts)
        maxIdx = findall(<(0), secondDerRts)

        max_x = maxminRts¹[maxIdx]
        min_x = maxminRts¹[minIdx]
        
        min_y = polyEq.(min_x)
        max_y = polyEq.(max_x)

        #Filter out precondition points ---------------------------------------------------------
        # Rules : ust be on negative side
        minIdx = findall(x -> x < 0, min_y)
        precon_x = min_x[minIdx]
        precon_y = min_y[minIdx]

        #Filter out budget values --------------------------------------------------------------
        # Rules : Must occure after threshold, must occur after precondition point
        if isempty(min_x)
                budget_x = Array([])
        else
                budget_x = zeroRts[zeroRts .> min_x[1]]
        end

        budgetIdx = findall(x -> x > timeThresh, budget_x)
        budget_x = budget_x[budgetIdx]

        # Filter delay --------------------------------------------------------------------------
        # Rules : Must occure before time threshold, must be on positive side 
        delayIdx1 = findall(x -> x < timeThresh, max_x)
        delayIdx2 = findall(x -> x > 0, max_y)
        delayIdx = findall(x -> x in delayIdx1, delayIdx2)
        delay_x = max_x[delayIdx]
        delay_y = max_y[delayIdx]

        # Filter Improvement decline point ------------------------------------------------------
        # Rules : Must be after threshold
        improvIdx = findall(x -> x > timeThresh, max_x)
        improvDecl_x = max_x[improvIdx]
        improvDecl_y = max_y[improvIdx]
        
        # Add the metrics into a dictionary
        metricsAll = Dict(
                "preconT" => precon_x,
                "preconV" => precon_y,
                "improvDeclT" => improvDecl_x,
                "improvDeclV" => improvDecl_y,
                "delayT" => delay_x,
                "delayV" => delay_y,
                "budgetT" => zeroRts,
        )

        metrics = Dict{String,Any}()
        for (k,v) in metricsAll
                if !isempty(v)
                    push!(metrics, k => Array{Any}([v[1]]))
                    if endswith(k,"T")
                        push!(metrics, k * "S" => Array{Date}([df.time[1] + convDateTime(v[1])]))
                    end
                else 
                    push!(metrics, k => Array{Any}([missing]))
                    if endswith(k,"T")
                        push!(metrics, k * "S" => Array{Any}([missing]))
                    end
                end
        end

        if !ismissing(metrics["budgetT"][1])
                push!(metrics, "budgetV" => Array{Float64}([0.0]))
        else
                push!(metrics, "budgetV" => Array{Any}([missing]))
        end

        
        push!(metrics, "id" => unique(df.id))
        push!(metrics, "time" => df.time)
        push!(metrics, "monthsSinceErup" => x)
        push!(metrics, "targetName" => target)
        push!(metrics, "target" => df[:,3])  
        push!(metrics, "fit" => df.yhat)
        push!(metrics, "ployEq" => Array([polyEq]))
        push!(metrics, "polyEqDer1" => Array([polyEq¹]))
        push!(metrics, "polyEqDer2" => Array([polyEq²]))
        

        return metrics
end

# Function to plot anomaly ---------------------------------------------

function anomPlot(metrics::Dict)
        theme(:bright)
        gr()
        p1 = plot(
                metrics["time"],
                metrics["target"],
                line = (:dodgerblue,3),
                label = "CDI - actual",
                xlabel = "Date",
                ylabel = "CDI",
                title = metrics["id"],
                titlelocation = :left
        )

        p2 = plot(
                metrics["monthsSinceErup"],
                metrics["fit"],
                line = (:slategray,3),
                label = "CDI - polynomial fit",
                xlabel = "Months since impact",
                ylabel = "CDI"
        )

        if !ismissing(metrics["improvDeclT"][1])
                p2 = plot!(
                        metrics["improvDeclT"],
                        metrics["improvDeclV"],
                        linetype = :scatter,
                        marker = (:circle,:limegreen, 5),
                        label = "Improvement decline Pt"
                )
        end

        if !ismissing(metrics["preconT"][1])
                p2 = plot!(
                        metrics["preconT"],
                        metrics["preconV"],
                        linetype = :scatter,
                        marker = (:circle, :coral,5),
                        label = "precondition state"
                )
        end

        if !ismissing(metrics["delayT"][1])
                p2 = plot!(
                        metrics["delayT"],
                        metrics["delayV"],
                        linetype = :scatter,
                        marker = (:circle, :fuchsia, 5),
                        label = "delay"
                )
        end

        if !ismissing(metrics["budgetT"][1])
                p2 = plot!(
                        metrics["budgetT"],
                        metrics["budgetV"],
                        linetype = :scatter,
                        marker = (:circle, :dodgerblue,5),
                        label = "budget"
                )
        end

        p = plot(p1,p2, layout = (2,1), size = (800,600), legend = :outertopright)
end

end









