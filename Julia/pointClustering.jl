using SQLite
using DataFrames
using ScikitLearn: fit!,predict,@sk_import
using Gadfly

set_default_plot_size(10cm,10cm)
@sk_import cluster:KMeans

# Data
# ____________________________________________________________________________________
# Connecting to Database 

db = SQLite.DB("D:/GEE_Project/Databases/database.db")
tbls = SQLite.tables(db)

points = DataFrame(SQLite.DBInterface.execute(db, "SELECT * FROM points"))

# Select Cordon Caulle eruption
cc = points[points.eruption .== "CC2011", :]

# Select Feature
# ____________________________________________________________________________________

# Explore Features
begin
    selectedFeatures = Array{String}(["id","lat","lon","distance", "heading","elevation", "slope", "aspect", "CHILI"])
	df = select(cc, selectedFeatures)
	X = select(df,Not([:id,:lat,:lon]))
end

# Any missing values ?
describe(df, :nmissing)

# KMeans Model 
# ____________________________________________________________________________________

function kmeansModel(data::DataFrame,noClus::Int64)
    model = KMeans(n_clusters = noClus)
    fit!(model, convert(Matrix, data))

    return model
end

#Assess best number of clusters

begin
    ssd = Array{Float64}(undef,19)
    for i = 2:20
        ssd[i-1] = kmeansModel(df, i).inertia_
    end
    ssd
end

# Elbow plot 

let
    set_default_plot_size(18cm,10cm)
    df = DataFrame()
    df[:,:number_of_clusters] = collect(2:20)
    df[:,:inertia] = ssd
    plot(df, x = :number_of_clusters , y = :inertia, Geom.line, Geom.point)
end

# Plots to check classes
# ____________________________________________________________________________________

# Selected 5 clusters
df[:,:predicted_labels] = kmeansModel(df,5).labels_

begin
    set_default_plot_size(15cm,15cm)
    fig = plot(df, x=:lat, y= :lon, color = :predicted_labels, Geom.point)
end

# Distribution plot

function plothist(df::DataFrame, xvals::Symbol; colVals::Symbol = :predicted_labels)
    plot(df, x = xvals, color = colVals, Geom.histogram)
end

plothist(df,:elevation)