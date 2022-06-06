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

def tab2_layout(file_name,summary_dict,dfl_dict,dfa_dict):
    avl=[1]
    if file_name is None :
        
        return html.Div([
    
    html.Div([
                
         dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='plot_dependent',
                                options=[{'label': i, 'value': i} for i in avl],
                                placeholder = 'ddd',
                                persistence = True
                                )])),
   
             
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='plot_independent',
                                options=[{'label': i, 'value': i} for i in avl],
                                placeholder = 'ddd',
                                persistence = True
                                )])),

                 ])]),
     


 

           html.Div([
                   
                   dcc.Graph(
                           
                     id='crossfilter-indicator-scatter'
                     )], style={'display': 'inline-block',
                                'width' : '49%'}),
        
        
           html.Div([
                                      
                   dcc.Graph(
                     
                     id='crossfilter-indicator-scatter2'
                     )], style={'display': 'inline-block',
                                'width' : '49%',}),
      
       
          html.Div([
        
                   dash_table.DataTable(
                   id='datatable-interactivity2',
                   editable=True,
                   filter_action="native",
                   sort_action="native",
                   sort_mode="multi",
#                  column_selectable="single",
#                  row_selectable="multi",
#                  row_deletable=True,
#                  selected_columns=[],
#                  selected_rows=[],
                   page_action="native",
                   ) 
           
            ], style={'padding': '30px 5px'})
     ])


#=========================for file name not nonetype

    else:
        #dfl_df=pd.DataFrame(dfl_dict)
        dfa_df=pd.DataFrame(dfa_dict)
        avl=dfa_df.columns
        print(file_name['props']['children'])
        return html.Div([
    
    html.Div([
                
         dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='plot_dependent',
                                options=[{'label': i, 'value': i} for i in avl],
                                placeholder = 'ddd',
                                persistence = True
                                )])),
   
            html.Br(), 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='plot_independent',
                                options=[{'label': i, 'value': i} for i in avl],
                                placeholder = 'ddd',
                                persistence = True
                                )])),

                 ])]),
     


 

           html.Div([
                   
                   dcc.Graph(
                           
                     id='crossfilter-indicator-scatter'
                     )], style={'display': 'inline-block',
                                'width' : '100%'}),
        
        
           
      
       
          html.Div([
        
                   dash_table.DataTable(
                   id='datatable-interactivity2',
                   style_cell={'backgroundColor': 'rgb(8, 8, 8)','color': 'white','text-align':'center'},
                   style_header={'backgroundColor': 'rgb(22, 62, 122)','fontWeight': 'bold'},
                   editable=True,
                   filter_action="native",
                   sort_action="native",
                   sort_mode="multi",
#                  column_selectable="single",
#                  row_selectable="multi",
#                  row_deletable=True,
#                  selected_columns=[],
#                  selected_rows=[],
                   page_action="native",
                   ) 
           
            ], style={'padding': '30px 5px'})
     ])
