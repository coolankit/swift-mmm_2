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

def build_top_panel(contri_data,contri_cols):
    print("Inside Post Model Tab")
    print("Printing value passed")
    print(contri_cols)
    return html.Div( id='upload_div',
                    children=[
    html.Label(id="metric-select-title", children="Upload Excel file which contains 3 tabs "),
    html.Br(),

    
    #html.Div(id='output-data-upload'),
    html.Div(
        id="top-section-container",
        className="row",
        children=[
                
            # Metrics summary
            html.Div(
                id="metric-summary-session",
                className="eight columns",
                children=[
                    
                    generate_section_banner("Contribution Table"),
                    html.Div(
                        id="metric-div",
                        children=[
                                
                                
                            #generate_metric_list_header(),
                            html.Div(
                                id="metric-rows",
                                children=[
                                        
                                     dash_table.DataTable(id='contribution_table',
                                     columns=contri_cols,
                                     data=contri_data,                                   
                                     editable=True,                                     
                                     style_cell={'backgroundColor': 'rgb(8, 8, 8)','color': 'white','text-align':'center'},
                                     style_header={'backgroundColor': 'rgb(22, 62, 122)','fontWeight': 'bold'}
                                     ),                           
                                ],
                                
                            ),
                           html.Div(id='button_div',
                                    children=[
                                            
                                   html.Button(id="update_contri_button",children="Click here to update table",                     
                                   style={"backgroundColor":"rgb(67,207,229)","color":"black"}),
                                               
                                   html.Button(id="update_decomp_chart",children="Update Decomp. Chart",                     
                                   style={"backgroundColor":"rgb(67,207,229)","color":"black"})
                                   ],
                                               
                                   ), 
                          html.Br(),
                          
                                
                        ],
                    ),
                ],
            ),


        ],
    ),
    html.Br(),
    html.P("Decomposition - Section"),
    #no division visible from here
    #Decomp div container starts from here
    generate_section_banner("Decomposition - by Percentage "),
    html.Div(id='download_div'),
    
    html.Div(id='metric-rows', 
             className='row',
             children=[
            html.Br(),
            
            html.Div(id='decomp_div',children=[
                    
                    
                    dash_table.DataTable(id='decomp_absolute_table',
                                     #columns=([{'id': p, 'name': p} for p in col_names]),
                                     #data=generate_table_data(),                                   
                                     editable=True,                                     
                                     style_cell={'backgroundColor': 'rgb(8, 8, 8)','color': 'white','text-align':'center'},
                                     style_header={'backgroundColor': 'rgb(22, 162, 122)','fontWeight': 'bold'}
                                     ),
                    
                    
                    
                    
                    ],),
                                  
                                  
                                  
                                  
                                  ],),
    html.Br(),
    html.Br(),
                
    html.Div(id='control-chart-container',children=[dcc.Graph(id='decomp_chart'),]),
    
    
    ],  
    
    )

def generate_section_banner(title):
    return html.Div(className="section-banner", children=title)
                                    