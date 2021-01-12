import dash 
import dash_core_components as dcc 
import dash_html_components as html 
from dash.dependencies import Input, Output 
import plotly.graph_objs as go 
import plotly.express as px

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

# Fits
# _____________________________________________________________________________________
fitsColVals = fits.columns.values[[i.startswith(("LSSR","MODIS")) for i in fits.columns.values]]

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
        )
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
                        id = "tsPlot-id"
                    )
                ]
        ),
        html.Div([
                dcc.Graph(
                    id = "anomalyPlot-id"
                )
            ]
        )


    ],style = {"width" : "48%", "float" : "right", "display" : "inline-block"}
    )

])

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
            marker = go.scattermapbox.Marker(size = 8, color = "dodgerblue"),
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

#Time series
# ------------------------------------------------------------------------------------------------------------


@app.callback(
    Output("tsPlot-id", "figure"),
    [Input("fits-id","value"),
    Input("map-id","clickData")]
)

def plotTS(fitsVal,clickVal):
    selectedId = str(clickVal["points"][0]["customdata"])
    df = fits[fits.id == selectedId]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = df.time, 
        y = df[fitsVal],
        mode = "lines",
        line = dict(color = "dodgerblue")
    ))
    fig.update_layout(
        xaxis_title = "Dates", yaxis_title = "Ndvi/Evi", template = "plotly_white", title = "Fitted TS"
    )

    return fig

#Anomaly
# -----------------------------------------------------------------------------------------------------------
@app.callback(
    Output("anomalyPlot-id", "figure"),
    [Input("anomaly-id","value"),
    Input("fits-id", "value"),
    Input("map-id", "clickData")]
)

def plotAnomaly(anomalyVal,fitsVal,clickVal):
    selectedId = str(clickVal["points"][0]["customdata"])
    df = anomalyM[anomalyM.id == selectedId]
    target = fitsVal + "." + anomalyVal
    print(target)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = df.time, 
        y = df[target],
        mode = "lines",
        line = dict(color = "coral")
    ))
    fig.update_layout(
        xaxis_title = "Dates (Post Eruption)", yaxis_title = "Ndvi/Evi", template = "plotly_white", title = "Anomaly"
        )
    return fig

if __name__ == "__main__":
    app.run_server(debug = False)
   

