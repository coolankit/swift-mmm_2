import pandas as pd
import re
import numpy as np
import statsmodels.formula.api as sm
import scipy.special as ssp

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import statsmodels.api as smm

from functions_used import summary_stats
from functions_used import contribution


#df= pd.read_excel(r'try_df.xlsx')
#summary, dfl, dfa = summary_stats(df,'weekly')
#df =dfl
#available_indicators = df.columns

def tab1_layout(file_name,summary_dict,dfl_dict,dfa_dict,params_col): 
#    if len(file_name['props']['children'])>0 :
     available_indicators=[1] 
     if file_name is None :
        
        return html.Div(children=[
        html.Div(children=[
                
         dbc.Row([
                 
            dbc.Col(
                
                html.Div(children=[
                        dcc.Dropdown(
                                id='dependent-column',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Dependent Variable',
                                persistence = False,
                                #style={'width':'50%',},
                                persistence_type = 'session'
                                )])),
             
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='base',
                                options=[{'label': 'standard', 'value': 1},
                                         {'label': '1-standard', 'value': 2},
                                         {'label': 'drop from average', 'value': 3}],
                                placeholder = 'baseline formula',
                                value = 1,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),

                 ]),

        dbc.Row([
                 
            dbc.Col(
                
                html.Div(children=[
                        dcc.Dropdown(
                                id='tv-column',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Independent Variable',
                                value = 'null',
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l1',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),

                 ]),
         
        dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var1',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'check',
                                value = 'null',
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l2',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),

                 ]),

       dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var2',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Dependent Variable',
                                value = 'null',
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l3',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),

                 ]),
       
       dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var3',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Dependent Variable',
                                value = 'null',
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l4',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),

                 ]),

       dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var4',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Dependent Variable',
                                value = 'null',
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l5',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),

                 ]),
             
     
    dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var5',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Dependent Variable',
                                value = 'null',
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l6',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),

                 ]),
    
    dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var6',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Dependent Variable',
                                value = 'null',
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l7',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),

                 ]),
    
    dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var7',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Dependent Variable',
                                value = 'null',
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l8',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),

                 ]),
    dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var8',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Dependent Variable',
                                value = 'null',
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l9',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),

                 ]),
    dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var9',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Dependent Variable',
                                value = 'null',
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l10',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),

                 ]),
    dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var10',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Dependent Variable',
                                value = 'null',
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l11',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),

                 ]),
    

          ], style = {'padding': '30px 5px'}),
    
#    html.Button(
#                        id="regression_button", children="Run Regression", n_clicks=0
#                    ),
#                   

    html.Div( 

                    [dash_table.DataTable(
                    id='datatable-interactivity')], 
            style={'padding': '40px 5px'}

             ),
   
    html.Div([
        
                   dash_table.DataTable(
                   id='correlation_matrix',
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
           
            ], 
            style={'padding': '40px 5px'}),
    
    html.Div(id='control-chart-container',children=[dcc.Graph(id='avp_graph')],),
            
            
            
   ])


        
        
 #===================for  file_name not NoneType===================
     else:
        #summary_df=pd.DataFrame(summary_dict)
        dfl_df=pd.DataFrame(dfl_dict)
        #dfa_df=pd.DataFrame(dfa_dict)
        available_indicators=dfl_df.columns
        #available_indicators.append('null')
        print("************************************************************************************************************")
        print(type(params_col))
        #params_df=pd.DataFrame(params_col)
        params_col = [ sub['name'] for sub in params_col ] 
        
        #params_col=params_df.columns
        #available_indicators=params_col
        print((params_col))
        params_col.append('null')
        params_col.append('null')
        params_col.append('null')
        params_col.append('null')
        params_col.reverse()
        #print(file_name['props']['children'])
        list_of_drop_down_elements=list()
             
 
        return html.Div(children=[
            html.Div(children=[
                
         dbc.Row([
                 
            dbc.Col(
                
                html.Div(children=[
                        dcc.Dropdown(
                                id='dependent-column',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Dependent Variable',
                                persistence = False,
                                #style={'width':'50%',},
                                persistence_type = 'session',
                                value = params_col.pop()
                                )])),
             
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='base',
                                options=[{'label': 'standard', 'value': 1},
                                         {'label': '1-standard', 'value': 2},
                                         {'label': 'drop from average', 'value': 3}],
                                placeholder = 'baseline formula',
                                value = 1,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),

                 ]),

        dbc.Row([
                 
            dbc.Col(
                
                html.Div(children=[
                        dcc.Dropdown(
                                id='tv-column',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Independent Variable',
                                #value = 'null',
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session',
                                value = params_col.pop()
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l1',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),

                 ]),
         
        dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var1',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'check',
                               # value = 'null',
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session',
                                value = params_col.pop()
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l2',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session',
                                )])),

                 ]),

       dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var2',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Dependent Variable',
                                #value = 'null',
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session',
                                value = params_col.pop()
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l3',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session',
                                )])),

                 ]),
       
       dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var3',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Dependent Variable',
                                #value = 'null',
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session',
                                value = params_col.pop()
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l4',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),

                 ]),

       dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var4',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Dependent Variable',
                                #value = 'null',
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session',
                                value = params_col.pop()
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l5',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),

                 ]),
             
     
    dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var5',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Dependent Variable',
                                #value = 'null',
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session',
                                value = params_col.pop()
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l6',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),

                 ]),
    
    dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var6',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Dependent Variable',
                                #value = 'null',
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session',
                                value = params_col.pop()
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l7',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),

                 ]),
    
    dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var7',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Dependent Variable',
                                #value = 'null',
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session',
                                value = params_col.pop()
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l8',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),

                 ]),
    dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var8',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Dependent Variable',
                                #value = 'null',
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session',
                                value = params_col.pop()
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l9',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),

                 ]),
    dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var9',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Dependent Variable',
                                #value = 'null',
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session',
                                value = params_col.pop()
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l10',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),

                 ]),
    dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var10',
                                options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Dependent Variable',
                                #value = 'null',
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session',
                                value = params_col.pop()
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l11',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                #style={'width':'50%',},
                                persistence = False,
                                persistence_type = 'session'
                                )])),

                 ]),

          ]),
    
#        html.Button(
#                        id="regression_button", children="Run Regression", n_clicks=0
#                    ),
         
    
        html.Br(),
        generate_section_banner("Contribution Table"),
        
        
        
        html.Div( 
            
    
                        [dash_table.DataTable(
                        id='datatable-interactivity',
                        style_cell={'backgroundColor': 'rgb(8, 8, 8)','color': 'white','text-align':'center'},
                        style_header={'backgroundColor': 'rgb(22, 62, 122)','fontWeight': 'bold'})], 
                
    
                 ),
       html.Button(id="freeze_contribution",children="Click to freeze this contribution table",                     
                   style={"backgroundColor":"rgb(67,207,229)","color":"black"}),
            
       html.Br(),
       html.Div(id='control-chart-container',children=[dcc.Graph(id='avp_graph')],),
     
       generate_section_banner("Correlation Matrix"),
       html.Div(id="metric-rows",children=[
            
                       dash_table.DataTable(
                       id='correlation_matrix',
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
               
                ], 
                )
                
                
                
       ])

def generate_section_banner(title):
    return html.Div(className="section-banner", children=title)
