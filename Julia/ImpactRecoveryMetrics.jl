using DataFrames
using Query
#using Plots
using Polynomials: fit,derivative,roots
using CSV
using Dates
using Gadfly

# Function to convert months in floats to Dates : Used for polynomial roots
function convDateTime(x)
        fm,m = modf(x)
        Day(Int(round(30.416 * fm))) + Month(Int(m))
end

# Function to compute anomaly metrics

function anomMet(data::DataFrame, idNo::Any, target::String ; uniqIdIdx::Int64 = 1,timeThresh::Int64 = 12)
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

        #Filter out precondition points : Precondition points must be negative
        minIdx = findall(x -> x < 0, min_y)
        min_x = min_x[minIdx]
        min_y = min_y[minIdx]

        #Filter zeroRts : must happen after a minima
        if isempty(min_x)
                zeroRts = Array([])
        else
                zeroRts = zeroRts[zeroRts .> min_x[1]]
        end

        # Filter delay
        delayIdx = findall(x -> x < timeThresh, max_x)
        delay_x = max_x[delayIdx]
        delay_y = max_y[delayIdx]

        # Filter Improvement
        improvIdx = findall(x -> x > timeThresh, max_x)
        improv_x = max_x[improvIdx]
        improv_y = max_y[improvIdx]
        
        # Add the metrics into a dictionary
        metricsAll = Dict(
                "preconT" => min_x,
                "preconV" => min_y,
                "improvDeclT" => improv_x,
                "improvDeclV" => improv_y,
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

        push!(metrics, "id" => unique(df.id))
        push!(metrics, "time" => df.time)
        push!(metrics, "monthsSinceErup" => x)
        push!(metrics, "targetName" => target)
        push!(metrics, "target" => df[:,3])  
        push!(metrics, "fit" => df.yhat)
        push!(metrics, "ployEq" => Array([polyEq]))
        push!(metrics, "polyEqDer1" => Array([polyEq¹]))
        push!(metrics, "polyEqDer2" => Array([polyEq²]))
        if !ismissing(metrics["budgetT"][1])
                push!(metrics, "budgetV" => Array{Float64}([0.0]))
        else
                push!(metrics, "budgetV" => Array{Any}([missing]))
        end

        return metrics
end

# Function to plot anomaly ---------------------------------------------

function anomPlot(metrics::Dict)
        theme(:wong)
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

function anomPlot2(met::Dict)
        set_default_plot_size(20cm,18cm)
        p1 = plot(
                x = met["time"], 
                y = met["target"], 
                Geom.line, 
                Theme(
                        line_width = 1mm
                        ,default_color = colorant"dodgerblue"
                ),
                Guide.xlabel("Dates"),
                Guide.ylabel(met["targetName"]),
                Guide.title(met["id"][1])
        )

        p2 = plot(
                layer(
                        x = met["monthsSinceErup"],
                        y = met["fit"],
                        Geom.line,
                        style(
                                line_width = 1mm,
                                default_color = colorant"slategray"
                        )
                ),
                layer(
                        x = met["delayT"],
                        y = met["delayV"],
                        Geom.point,
                        style(
                                point_size = 1.75mm,
                                default_color = colorant"fuchsia"
                        )       
                ),
                layer(
                        x = met["preconT"],
                        y = met["preconV"],
                        Geom.point,
                        style(
                                point_size = 1.75mm,
                                default_color = colorant"coral"
                        )
                ),
                layer(
                        x = met["improvDeclT"],
                        y = met["improvDeclV"],
                        Geom.point,
                        style(
                                point_size = 1.75mm,
                                default_color = colorant"limegreen"
                        )
                ),
                layer(
                        x = met["budgetT"],
                        y = met["budgetV"],
                        Geom.point,
                        style(
                                point_size = 1.75mm,
                                default_color = colorant"steelblue"
                        )
                ),
                Guide.xlabel("Time since eruption (months)"),
                Guide.ylabel(met["targetName"]),
                Guide.manual_color_key("Legend",["delay","precondition","budget","improvement decline pt"],["fuchsia","coral","steelblue","limegreen"]), 
                Theme(key_position = :inside)
        ) 

        p = vstack(p1,p2)
end
#
using SQLite
# Connect to Database
db = SQLite.DB("D:/GEE_Project/Databases/database.db")
anomalyM = DataFrame(SQLite.DBInterface.execute(db, "SELECT * FROM anomalyM"))

met = anomMet(anomalyM,NaN, "LSSR.EVI.CDI",;uniqIdIdx = 60)
met

anomPlot2(met)
#p = anomPlot(met)

let 
        
end



