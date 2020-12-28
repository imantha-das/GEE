using DataFrames
using Query
using Plots
using Polynomials: fit,derivative,roots
using CSV
using Dates


# Function to compute anomaly metrics

function anomMet(data::DataFrame, idNo::Any, target::String; uniqIdIdx::Int64 = 1,timeThresh::Int64 = 12)
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

        # Filter delay
        delayIdx = findall(x -> x < timeThresh, max_x)
        delay_x = max_x[delayIdx]
        delay_y = max_y[delayIdx]

        # Filter Improvement
        improvIdx = findall(x -> x > timeThresh, max_x)
        improv_x = max_x[improvIdx]
        improv_y = max_y[improvIdx]
        

        metrics = Dict(
                "precon_x" => min_x,
                "precon_y" => min_y,
                "improv_x" => improv_x,
                "improv_y" => improv_y,
                "delay_x" => delay_x,
                "delay_y" => delay_y,
                "budget" => zeroRts
        )

        return df,metrics
end

# Function to plot anomaly ---------------------------------------------

function anomPlot(df::DataFrame,metrics::Dict)
        theme(:wong)
        gr()
        p1 = plot(
                df.time,
                df[:,3],
                line = (:dodgerblue,3),
                label = "CDI - actual",
                xlabel = "Date",
                ylabel = "CDI",
                title = unique(df.id)[1],
                titlelocation = :left
        )

        p2 = plot(
                df.months_since_impact,
                df.yhat,
                line = (:slategray,3),
                label = "CDI - polynomial fit",
                xlabel = "Months since impact",
                ylabel = "CDI"
        )

        if length(metrics["improv_x"]) > 0
                p2 = plot!(
                        metrics["improv_x"],
                        metrics["improv_y"],
                        linetype = :scatter,
                        marker = (:circle,:limegreen, 5),
                        label = "Improvement"
                )
        end

        if length(metrics["precon_x"]) > 0
                p2 = plot!(
                        metrics["precon_x"],
                        metrics["precon_y"],
                        linetype = :scatter,
                        marker = (:circle, :coral,5),
                        label = "precondition state"
                )
        end

        if length(metrics["delay_x"]) > 0
                p2 = plot!(
                        metrics["delay_x"],
                        zeros(length(metrics["delay_x"])),
                        linetype = :scatter,
                        marker = (:circle, :fuchsia, 5),
                        label = "delay"
                )
        end

        if length(metrics["budget"]) > 0
                p2 = plot!(
                        metrics["budget"],
                        zeros(length(metrics["budget"])),
                        linetype = :scatter,
                        marker = (:circle, :dodgerblue,5),
                        lable = "budget"
                )
        end

        p = plot(p1,p2, layout = (2,1), size = (600,600), legend = :bottomleft)
end

#=
df,met = anomMet(data,NaN, "LSSR.EVI.CDI";uniqIdIdx = 54)
p = anomPlot(df,met)
=#


