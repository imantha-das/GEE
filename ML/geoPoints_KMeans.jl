using DataFrames
using Plots
using ScikitLearn: fit!,predict,@sk_import
using SQLite
theme(:bright)

push!(LOAD_PATH, "D:/Julia/ee/Anomaly_Computation")
using ImpactRecoveryMetrics: anomMet, anomPlot

@sk_import cluster:KMeans

# Load Tables from DataFrame 
# ------------------------------------------------------------------------------------------------

# Connect to DataBase
function con2db(pathName::String, dbName::String)::DataFrame
    db = SQLite.DB(pathName)
    q = "SELECT * FROM" * " " * dbName
    df = DataFrame(SQLite.DBInterface.execute(db,q))
    return df
end

pts = con2db("D:/GEE_Project/Databases/database.db", "points")
anomalyM = con2db("D:/GEE_Project/Databases/database.db","anomalyM")

# Select point for CC2011 eruption
# --------------------------------------------------------------------------------------------------

# Filtering eruption CC2011
merapiPts = pts[pts.eruption .== "Merapi2010",:]

# UniqIds for CC2011
merapiUniqIds = unique(merapiPts.id)


# Extract Polyfits 
# -------------------------------------------------------------------------------------------------

function compPolyfits(uniqIds::Array{String},dataset::String)::DataFrame
    polyfitDfArr = Array{DataFrame}([])
    for id in uniqIds
        
        met = anomMet(anomalyM,id,dataset)
        polyfitDf = DataFrame(Dict(:id => repeat(met["id"], length(met["fit"])),:time => met["time"], :monthSininceErup => met["monthsSinceErup"], :fit => met["fit"]))

        push!(polyfitDfArr, polyfitDf)
    end 
    return polyfit = vcat(polyfitDfArr...)
end

polyfits = compPolyfits(merapiUniqIds, "LSSR.EVI.CDI")


# Kmeans 
# -----------------------------------------------------------------------------------------------

# Select Features : All numerical features
names(pts)
X = select(merapiPts, [:id,:distance,:heading,:elevation,:slope,:aspect,:CHILI,:mass,:mode,:sorting,:deposit_field])

# Drop any missing values
dropmissing!(X)

# Kmeans clustering with 5 clusters
km5 = KMeans(n_clusters = 5)
fit!(km5, convert(Matrix,select(X,Not(:id))))

X[:,"labels"] = km5.labels_

# Plot anomaly fits - single point

begin 
    p = plot()
    colorList = Array{String}(["dodgerblue","coral","fuchsia","slategray","limegreen"])
    for id in unique(X.id)
        polyfit = polyfits[polyfits.id .== id,:]
        lab = X[X.id .== id,:][:,:labels][1] + 1

        @show lab
        @show colorList[lab]

        p = plot!(
            polyfit.time,
            polyfit.fit,
            line = (cgrad(:tab10,5, categorical = true)[lab]),
            legend = false
        )
    end

    p
end

