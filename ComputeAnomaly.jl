using SQLite 
using DataFrames 
using Plots 
using Query 
using Dates
using Statistics 

theme(:bright)

# COnnect to SQL Database
conn = SQLite.DB("D:/GEE_Project/Databases/database.db")
q1 = "SELECT * FROM FITS"
res = DBInterface.execute(conn, q1)
fits = DataFrame(res)

function computeAnomaly(data::DataFrame, target::String, eDate::Date, noPoints::Int64)::DataFrame
    df = copy(data)
    uniqIds = unique(df.id)

    # Change any column names with . to _
    colNames = replace.(names(df), "." => "_")
    rename!(df, colNames)

    # Replace target name with . to _
    target = replace.(target, "." => "_")

    # Change time column to dateformat
    df.time = Date.(df.time, dateformat"Y-m-d H:M:S")

    # Empty darray to store df_post dataframes
    df_array = Array{Any}[]

    for id in uniqIds[1:noPoints]

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
    return df_post_concat
end

# function call
df = computeAnomaly(fits, "LSSR.NDVI", Date(2010,10,26),5)


