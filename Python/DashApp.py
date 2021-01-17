import dash 
import dash_core_components as dcc 
import dash_html_components as html 
from dash.dependencies import Input, Output 
import dash_table

import plotly.graph_objs as go 
from plotly.subplots import make_subplots

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from termcolor import colored

import credentials

# __________________________________________________________________________________

# connect to database
fname = "D:/GEE_project/Databases/database.db"
conn = sqlite3.connect(fname)
cur = conn.cursor()


#DELETE ME
'''gvp_eruptions gvp_events gvp_refs  LSSR      
MODISVI  MODISGPP MODISLC MODISLCD CHIRPS FITS SEAS TRENDS TaskList GLDAS pheno FITSst SEASst TRENDSst referenceM referenceY anomalyM anomalyY static points soil smoothParam scores eruption gvp_eruptions gvp_events gvp_refs

tables = cur.execute("SELECT * FROM SQLite_master").fetchall()
for tb in tables:
    print(tb[2])
'''
# Load tables from database
# _____________________________________________________________________________________
points = pd.read_sql(sql = "SELECT * FROM points", con = conn)
eruption = pd.read_sql(sql = "SELECT * FROM eruption", con = conn)
chirps = pd.read_sql(sql = "SELECT * FROM CHIRPS", con = conn)
gldas = pd.read_sql(sql = "SELECT * FROM GLDAS", con = conn)
anomalyM = pd.read_sql(sql = "SELECT * FROM anomalyM", con = conn)
fits = pd.read_sql(sql = "SELECT * FROM FITS", con = conn)
modisVI = pd.read_sql(sql = "SELECT * FROM MODISVI", con = conn)
modisGPP = pd.read_sql(sql = "SELECT * FROM MODISGPP", con = conn)
modisLC = pd.read_sql(sql = "SELECT * FROM MODISLC", con = conn)
modisLCD = pd.read_sql(sql = "SELECT * FROM MODISLCD", con = conn)


conn.close()

# Load tables from polyfitsMetrics.db
fname = "D:/GEE_project/Databases/polyfitsMetrics.db"
conn = sqlite3.connect(fname)
cur = conn.cursor()

polyfits = pd.read_sql(sql = "SELECT * FROM polyfits", con = conn)
metrics = pd.read_sql(sql = "SELECT * FROM metrics", con = conn)

conn.close()

# Get mapbox acces key
# _____________________________________________________________________________________
mapboxKey = credentials.mapbox_key

# Colors & Styles
# ______________________________________________________________________________________
styDict = {
    "fs_title" : "46px",
    "fc_title" : "olive",
    "fs_header" : "30px",
    'fc_header' : "darkslategray"
}

# Column names
# _____________________________________________________________________________________
fitsColVals = fits.columns.values[[i.startswith(("LSSR","MODIS")) for i in fits.columns.values]]
gldasColVals = [i for i in gldas.columns.values if not i in ["id","time","dataset","source"]]

ptsCols = points.columns.values
metCols = metrics.columns.values
ptsMetCols = np.concatenate((ptsCols,metCols), axis = 0)
ptsMetSelCols  = [i for i in ptsMetCols if not i in ["id","lat","lon","label","eruption","dataset","delayTS","preconTS","improvDeclTS", "budgetTS"]]


# Dash app
# _______________________________________________________________________________________

app = dash.Dash(__name__, prevent_initial_callbacks=True)

app.layout = html.Div([
    html.H1(
        children = "Earth observation data for volcanic impact assessment",
        style = {"textAlign" : "center", "color" : styDict["fc_title"], "font-size" : styDict["fs_title"]}
    ),

    # Left Div Elements
    html.Div([

        html.Div([
                html.H3(
                    children = "Select eruption",
                    style = {"color" : styDict["fc_header"]}
                )
            ]
        ),
        html.Div([
                dcc.RadioItems(
                    id = "eruption-id",
                    options = [
                        {"label" : "Merapi", "value" : "Merapi2010"},
                        {"label" : "Cordon Caulle", "value" : "CC2011"},
                        {"label" : "Shinmoedake", "value" : "Shinmoedake2011"},
                        {"label" : "Kelude", "value" : "Kelud2014"},
                        {"label" : "Sinabung", "value" : "Sinabung2014"}
                    ],
                    value = "Merapi2010"
                )
            ]
        ),
        html.Div([
                dcc.Graph(
                    id = "map-id",
                    config = {"displayModeBar" : False, "scrollZoom" : True},
                )
            ]
        ),
        html.Div([
                dash_table.DataTable(
                    id = "scoreTable-id",
                    columns = [{"name" : i, "id" : i} for i in metrics.columns if not i in ["delayT", "preconT", "improvDeclT", "budgetT"]],
                )
        ]),

        html.Br(),
        html.Br(),
        html.Br(),

        html.Div([
                dcc.Dropdown(
                    id = "corr-id-1",
                    options = [{"label" : i, "value" : i} for i in ptsMetSelCols],
                    value = ptsMetSelCols[0],
                    style = {"width" : "48%"}
                ),
                
                dcc.Dropdown(
                    id = "corr-id-2",
                    options = [{"label": i, "value" : i} for i in ptsMetSelCols],
                    value = ptsMetSelCols[1],
                    style = {"width" : "48%"}
                )
        ]),
               
       
        html.Div([
                dcc.Graph(
                    id = "corr-plot-id"
                )
        ])

    ], style = {"width" : "48%", "float" : "left", "display" : "inline-block"}
    ),

    # Right Div Elements
    html.Div([
        
        html.Div([
                dcc.Dropdown(
                    id = "fits-id",
                    options = [{"label" : i , "value" : i} for i in fitsColVals],
                    value = fitsColVals[0]
                )
            ]
        ),
        html.Div([
                dcc.RadioItems(
                    id = "anomaly-id",
                    options = [
                        {"label" : "CDI" , "value" : "CDI"},
                        {"label" : "SVI", "value" : "SVI"},
                        {"label" : "VCI", "value" : "VCI"},
                        {"label" : "VPI", "value" : "VPI"}
                    ],
                    value = "CDI"
                )
            ]
        ),
        html.Div([
                dcc.Graph(
                        id = "ts-anom-plot-id",
                    )
                ]
        ),
        
        html.Div([
                dcc.RadioItems(
                    id = "rawData-ri-id",
                        options = [
                            {"label" : "MODISVI-NDVI", "value" : "MODISVI-NDVI"},
                            {"label" : "MODISVI-EVI", "value" : "MODISVI-EVI"},
                            {"label" : "MODISGPP", "value" : "MODISGPP"},
                            {"label" : "MODISLC", "value" : "MODISLC"}
                        ],
                        value = "MODISVI-NDVI"
                )
        ]),

        html.Div([
                dcc.Graph(
                    id = "rawData-plot-id"
                )
        ]),

        html.Div([
                dcc.Dropdown(
                    id = "gldas-dd-id",
                    options = [
                        {"label" : i, "value" : i} for i in gldas.columns.values if not i in ["id","time","dataset","source"]
                    ],
                    value = gldas.columns.values[2]
                )       
        ]),
        
        html.Div([
                dcc.Graph(
                    id = "gldas-plot-id"
                )
        ])

    ],style = {"width" : "48%", "float" : "right", "display" : "inline-block"}
    )

])

#Callbacks
# ----------------------------------------------------------------------------------------------------

@app.callback(
    Output("map-id", "figure"),
    [Input("eruption-id", "value")]
)

def updateMap(erupVal):
    df = points[points.eruption == erupVal]

    if erupVal == "Merapi2010":
        volLat = -7.54
        volLon = 110.45
    elif erupVal == "CC2011":
        volLat = -40.58
        volLon = -72.11
    elif erupVal == "Shinmoedake2011":
        volLat = 31.91
        volLon = 130.88
    elif erupVal == "Kelud2014":
        volLat = -7.93
        volLon = 112.31
    elif erupVal == "Sinabung2014":
        volLat = 3.16
        volLon = 98.39

    fig = go.Figure()
    fig.add_trace(
        go.Scattermapbox(
            lat = df.lat,
            lon = df.lon,
            mode = "markers",
            marker = go.scattermapbox.Marker(size = 8, color = "indianred"),
            customdata = df.id
        )
    )

    fig.update_layout(
        uirevision = "foo",
        clickmode = "event+select",
        hovermode = "closest",
        hoverdistance = 1,
        mapbox = dict(
            accesstoken = mapboxKey,
            center = dict(lat = volLat, lon = volLon),
            pitch = 0,
            zoom = 8,
            style = "light"
            ),
        height = 750
        )

    return fig

#Time series & anomaly
# ------------------------------------------------------------------------------------------------------------

@app.callback(
    Output("ts-anom-plot-id", "figure"),
    [Input("fits-id","value"),
    Input("anomaly-id", "value"),
    Input("map-id","clickData")]
)

def plotTS(fitsVal, anomVal,clickVal):
    selectedId = str(clickVal["points"][0]["customdata"])
    
    target_anom = fitsVal + "." + anomVal
    target_polyfits = fitsVal + "." + anomVal + ".FIT"

    df_fits = fits[fits.id == selectedId]
    df_anom = anomalyM[anomalyM.id == selectedId]
    df_polyfits = polyfits[polyfits.id == selectedId]
    df_metrics = metrics[(metrics.id == selectedId) & (metrics.dataset == target_anom)]
    
    fig = make_subplots(rows = 2, cols = 1, subplot_titles = ["Time series", "Anomaly"])

    fig.add_trace(go.Scatter(
        x = df_fits.time, 
        y = df_fits[fitsVal],
        mode = "lines",
        line = dict(color = "slategray"),
        name = "Timeseries-Fit (Spline)"
    ), row = 1, col = 1)

    fig.add_trace(go.Scatter(
        x = df_anom.time, 
        y = df_anom[target_anom],
        mode = "lines",
        line = dict(color = "dodgerblue", dash = "dashdot", width = 1),
        name = "Anomaly"
    ), row = 2, col = 1)
    
    if anomVal == "CDI":
        fig.add_trace(go.Scatter(
            x = df_anom.time,
            y = df_polyfits[target_polyfits],
            mode = "lines",
            line = dict(color = "skyblue", width = 2),
            name = "Anomaly Fit (Polynomial)",
        ), row = 2, col = 1)

        # Add delay trace
        fig.add_trace(go.Scatter(
            x = df_metrics.delayTS,
            y = df_metrics.delayV,
            mode = "markers",
            marker = dict(
                color = "fuchsia",
                size = 10
            ),
            name = "delay"
        ), row = 2, col = 1)

        # Add precondition Trace
        fig.add_trace(go.Scatter(
            x = df_metrics.preconTS,
            y = df_metrics.preconV,
            mode = "markers",
            marker = dict(
                color = "coral",
                size = 10
            ),
            name = "precondition"
        ), row = 2, col = 1)

        #Add Improvement Decline point Trace
        fig.add_trace(go.Scatter(
            x = df_metrics.improvDeclTS,
            y = df_metrics.improvDeclV,
            mode = "markers",
            marker = dict(
                color = "limegreen",
                size = 10
            ),
            name = "Improvement decline pt"
        ), row = 2, col = 1)

        # Add budget
        fig.add_trace(go.Scatter(
            x = df_metrics.budgetTS,
            y = df_metrics.budgetV,
            mode = "markers",
            marker = dict(
                color = "steelblue",
                size = 10
            ),
            name = "budget"
        ), row = 2, col = 1)

    fig.update_layout(template = "plotly_white")
    fig.update_xaxes(title_text = "Dates", row = 1, col = 1)
    fig.update_yaxes(title_text = "Ndvi/Evi", row = 2, col = 1)

    return fig

# Raw data -----------------------------------------------------------------------------------------------

@app.callback(
    Output("rawData-plot-id", "figure"),
    [Input("rawData-ri-id", "value"),
    Input("map-id", "clickData")]
)

def plotRawData (satVal,clickVal):
    selectedId = str(clickVal["points"][0]["customdata"])
    print(colored(satVal,"red"))

    if satVal == "MODISVI-NDVI":
        data = modisVI[modisVI.id == selectedId]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x = data.time,
            y = data.NDVI,
            name = "MODIS-NDVI",
            mode = "markers+lines",
            line = dict(color = "slategray")
        ))

    elif satVal == "MODISVI-EVI":
        data = modisVI[modisVI.id == selectedId]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x = data.time,
            y = data.EVI,
            name = "MODIS-EVI",
            mode = "markers+lines",
            line = dict(color = "slategray")
        ))
    elif satVal == "MODISGPP":
        data = modisGPP[modisGPP.id == selectedId]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x = data.time,
            y = data.GPP,
            name = "Modis GPP",
            mode = "lines+markers",
            line = dict(color = "slategray")
        ))

    else:
        data = modisLC[modisLC.id == selectedId]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x = data.time,
            y = data.MODISLC,
            name = "Modis LC",
            mode = "lines+markers",
            line = dict(color = "slategray")
        ))
   
    fig.update_layout(xaxis_title = "time", yaxis_title = "ndvi/evi",template = "plotly_white", showlegend = True, title = "Raw Data")
    return fig


#Gldas
# ----------------------------------------------------------------------------------------------------------
@app.callback(
    Output("gldas-plot-id", "figure"),
    [Input("gldas-dd-id", "value"),
    Input("map-id", "clickData")]
)

def plotGldas(gldasVal,clickVal):
    selectedId = str(clickVal["points"][0]["customdata"])
    df = gldas[gldas.id == selectedId]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = df.time,
        y = df[gldasVal],
        mode = "lines+markers",
        line = dict(color = "slategray"),
        marker = dict(color = "slategray")
    ))
    fig.update_layout(
        xaxis_title = "Dates", yaxis_title = gldasVal, template = "plotly_white", title = "Environmental Timeseries"
        )
    return fig

# dash table
# -----------------------------------------------------------------------------------------------------------
@app.callback(
    Output("scoreTable-id","data"),
    [Input("fits-id","value"),
    Input("anomaly-id", "value"),
    Input("map-id","clickData")]
)

def scoreTable(fitsVal,anomVal, clickVal):
    selectedId = str(clickVal["points"][0]["customdata"])
    
    target_anom = fitsVal + "." + anomVal

    df_metrics = metrics[(metrics.id == selectedId) & (metrics.dataset == target_anom)]
    df_metrics.drop(["delayT", "preconT", "improvDeclT", "budgetT"],axis = 1, inplace = True)

    df_metrics[["delayV","preconV","improvDeclV","budgetV"]] =  df_metrics[["delayV","preconV","improvDeclV","budgetV"]].apply(lambda x: np.round(x,3))

    return df_metrics.to_dict("records")

# Corrolation plot 
# ----------------------------------------------------------------------------------------------------------

@app.callback(
    Output("corr-plot-id","figure"),
    [Input("eruption-id","value"),
    Input("corr-id-1", "value"),
    Input("corr-id-2", "value")]
)

def plotCorr(erupVal,corr1Val,corr2Val):
    ptsMet = pd.merge(points,metrics, on = "id")
    df = ptsMet[ptsMet.eruption == erupVal]

    '''
    fig = go.Figure(data = go.Splom(
        dimensions = [
            dict(label = corr1Val, values = df[corr1Val]),
            dict(label = corr2Val, values = df[corr2Val])
        ],
        marker = dict(color = "skyblue", size = 3)
    ))
    '''
    fig = go.Figure(go.Scatter(
        x = df[corr1Val],
        y = df[corr2Val],
        mode = "markers",
        marker = dict(color = "slategray", size = 3)
    ))

    fig.update_layout(xaxis_title = corr1Val, yaxis_title = corr2Val, template = "plotly_white", title = "Correlation")

    return fig

if __name__ == "__main__":
    app.run_server(debug = False)
    

