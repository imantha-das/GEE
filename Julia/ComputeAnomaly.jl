using DataFrames
using Query
using Dates
using Statistics
using Plotly 

# -----------------------------------------------------------------------------------------------------------------------------
#= Function to compute Anomaly
* Computes the anomaly by finding the difference between post eruption signal and pre eruption signal
* Computes the cumulative difference

Input arguments
* ptId : Point Id
    either give name of point 205936798913767 ...
    or set to NaN
* Data : Dataframe containing columns id, time and target column
* target : NDVI data
    i.e LSSR.NDVI
* eDate : Eruption date
    i.e : Date(2010,10,26)
* noPoints : optional argument
    set ptId to NaN if using this argument
    give the number of unique points that requires to be computed (more than one geographical point can be calculated)
=#

function computeAnomaly(ptId::Any,data::DataFrame, target::String, eDate::Date; noPoints::Int64 = 1)::DataFrame
    df = copy(data)

    # Change any column names with . to _
    colNames = replace.(names(df), "." => "_")
    rename!(df, colNames)

    # Replace target name with . to _
    target = replace.(target, "." => "_")

    # Change time column to dateformat
    df.time = Date.(df.time, dateformat"Y-m-d H:M:S")

    # Empty darray to store df_post dataframes
    df_array = Array{Any}[]

    # Select point Ids
    if typeof(ptId) == String
        uniqIds = Array{String}([ptId])
    else
        uniqIds = unique(df.id)[1:noPoints]
    end

    for id in uniqIds

        # Divide dataset into pre and post eruption
        df_pre = @from row in df begin
            @where (row.id == id) && (row.time < eDate)
            @select {id = row[:id], time = row[:time], target = row[Symbol(target)]}
            @collect DataFrame
        end

        df_post = @from row in df begin
           @where (row.id == id) && (row.time >= eDate)
           @select {id = row[:id], time = row[:time], target = row[Symbol(target)]}
           @collect DataFrame
        end

        # Extract the month column from date and
        df_pre[:,"month"] = month.(df_pre.time)
        df_post[:,"month"] = month.(df_post.time)

        # Compute the average month mean
        df_pre_avg = combine(groupby(df_pre, :month), "target" => mean)

        if sum(ismissing.(df_pre_avg.target_mean)) == 0

            # Allocate a column in df_post with NaN values to store pre_mean_avg
            df_post[:,"pre_mean"] .= NaN

            # Compute the pre-month-mean - post-month-mean
            for i = 1:12
                pre_mean_val = df_pre_avg[df_pre_avg.month .== i,:][:,:target_mean][1]
                df_post[df_post.month .== i, :pre_mean] .= pre_mean_val
            end

            # Compute difference
            df_post[:,"diff"] = df_post.target .- df_post.pre_mean

            #Compute cumulative some of difference
            df_post[:,"cumulative_diff"] = cumsum(df_post.diff)

            # Append df_post into an empty array
            push!(df_array, df_post)
        end
    end

    df_post_concat = reduce(vcat, df_array)
    df_post_concat = DataFrame(df_post_concat)
    rename!(df_post_concat, ["id","time",target,"month","pre_month_avg","diff","cumulative_diff"])
    #select!(df_post_concat, Not(:month))

    return df_post_concat
end

# -----------------------------------------------------------------------------------------------------------------------------
#= Function to plot Anomlay
* Plots Post eruption time series signal, diff and cumulative diff

Input argument
df : dataframe with columns time, post-eruption-target, diff, cumulative diff
=#

function plotAnom(data::DataFrame)
    use_style!(:gadfly_dark)
    uniqIds = unique(data.id)

    anomaly = Array{GenericTrace}([])
    preMeanAnomaly = Array{GenericTrace}([])
    diff = Array{GenericTrace}([])
    cumulative_diff = Array{GenericTrace}([])

    for id in uniqIds
        df = data[data.id .== id, :]
        trace1 = scatter(
            ;x = df.time,
            y = df[:,names(df)[3]],
            mode = "lines",
            name = String(id)
        )
        trace2 = scatter(
            ;x = df.time,
            y = df.diff,
            mode = "lines",
            name = String(id)
        )
        trace3 = scatter(
            ;x = df.time,
            y = df.cumulative_diff,
            mode = "line",
            name = String(id)
        )

        push!(anomaly,trace1)
        push!(diff,trace2)
        push!(cumulative_diff,trace3)
        
    end
    p1 = plot(anomaly, Layout(title = "postTS"))
    p2 = plot(diff, Layout(title = "Diff (postTS - preTS)"))
    p3 = plot(cumulative_diff, Layout(title = "Cumulative Diff (ΣDiff)"))
    
    return [p1,p2,p3]
end

function plotAnom2(data::DataFrame)
    uniqIds = unique(data.id)
    traceAnomaly = Array{GenericTrace}([])
    tracePreMean = Array{GenericTrace}([])
    traceDiff = Array{GenericTrace}([])
    
    for id in uniqIds
        df = data[data.id .== id, :]
        trace1 = scatter(
            x = df.time,
            y = df[:,names(df)[3]],
            mode = "lines",
            name = String(id)
        )
        print(typeof(trace1))
        push!(traceAnomaly,trace1)
    end

    data = [traceAnomaly]
    layout = Layout(title = "Anomaly", width = 2000)
    fig = Plotly.plot(traceAnomaly,layout)
end

# connect to database 
using SQLite
db = SQLite.DB("D:/GEE_Project/Databases/database.db")
tbls = SQLite.tables(db)
fitsTb = SQLite.DBInterface.execute(db,"SELECT * FROM FITS")
fits = DataFrame(fitsTb)

#function call
df = computeAnomaly(NaN,fits, "LSSR.NDVI", Date(2010,10,26), noPoints = 3)
print(first(df,5))
plotAnom2(df)


methods(Plotly.plot)

let 
    trace1 = scatter(
        x = collect(1:10),
        y = randn(10),
        mode = "lines",
        line = Dict(
            :color => "coral",
            :width => 3
            ),
        name = "red line"

    )
    trace2 = scatter(
        x = collect(1:10),
        y = randn(10),
        mode = "lines+markers",
        line = Dict(
            :color => "lightskyblue",
            :width => 3,
            ),
        marker = Dict(
            :color => "thistle",
            :size => 7.5
        ),
        name = "blue line"
    )

    layout = Layout(title = "Scatter Plot", xaxis_title = "x-axis", yaxis_title = "y-axis", showlegend = false, template = "plotly_white")

    data = [trace1,trace2]
    Plotly.plot(data,layout)

    typeof(data)
    
end

