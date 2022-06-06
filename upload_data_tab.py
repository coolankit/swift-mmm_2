import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
import dash_table
import pandas as pd
import re
import numpy as np
import statsmodels.formula.api as sm
import scipy.special as ssp

from functions_used import summary_stats
from functions_used import contribution


# =============================================================================
# df= pd.read_excel(r'try_df.xlsx')
# summary, dfl, dfa = summary_stats(df,'weekly')
# df =dfl
# available_indicators = df.columns
# avl = dfa.columns
# #summary_df=pd.DataFrame(summary_dict)

        
# =============================================================================
col_names=['Variable','Coeff.','logit avg.','logit min.','y pred. logit','y pred. min','y pred act.','y pred. act. min','% diff','Normalized Contri.']

def build_top_panel(contri_data,contri_cols):

    return html.Div( id='upload_div',
                    children=[
    html.Label(id="metric-select-title", children="Upload Input file here..."),
    html.Br(),

    
    dcc.Upload(id='upload-data',multiple=True,children=html.Div(['Drag and Drop or ',html.A('Select Files')])),
    dcc.Loading(id="loading-1",type="default",children=html.Div(id="loading-output-1")),
    #html.Div(id='output-data-upload'),
    html.Br(),
    html.H3(id='upload_response_div',children=[],style={'color':'rgb(51,255,255)'}),
    html.Div(id='user_manual',children=[html.Ul(children=[
            html.Li('If you have uploaded results of Automodeller switch to Post Model Dashboard straight away'),
            html.Li('Switch to Hypothesis Builder for altering existing hypotheis or building from scratch'),
            html.Li('Switch to "Trend Analysis" as aid while using Hypothesis Builder'),
             ]),]),
    

    
    ],  

    
    )
def generate_table_data():
    
    #pd.read_excel(os.path.join(APP_PATH, os.path.join("data", "")))
    
    return [dict({col: 2 for col in col_names})for i in range(1,2 )]
  
def generate_section_banner(title):
    return html.Div(className="section-banner", children=title)
                                    