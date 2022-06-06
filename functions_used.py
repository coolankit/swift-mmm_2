import pandas as pd
import re
import numpy as np
import statsmodels.formula.api as sm
import scipy.special as ssp
#import scipy.special
#import sklearn.preprocessing



#df = pd.read_excel(r'try_df.xlsx')

adstock_prefix_conditions = ['tv','ooh','print']

adstock_suffix_conditions = []



monthly = {'adstock_1':['adstock_1',0.0625], 'adstock_2':['adstock_2',0.25], 'adstock_3':['adstock_3',0.3969],
                 'adstock_4':['adstock_4',0.5], 'adstock_5':['adstock_5',0.5743], 'adstock_6':['adstock_6',0.63],
                 'adstock_7':['adstock_7',0.6730], 'adstock_8':['adstock_8',0.7071]}

weekly = {'adstock_1':['adstock_1',0.5], 'adstock_2':['adstock_2',0.71], 'adstock_3':['adstock_3',0.79],
                'adstock_4':['adstock_4',0.84], 'adstock_5':['adstock_5',0.87], 'adstock_6':['adstock_6',0.89],
                 'adstock_7':['adstock_7',0.91], 'adstock_8':['adstock_8',0.92]}


daily = {'adstock_1':['adstock_1',0.91], 'adstock_2':['adstock_2',0.95], 'adstock_3':['adstock_3',0.97],
                 'adstock_4':['adstock_4',0.98], 'adstock_5':['adstock_5',0.98], 'adstock_6':['adstock_6',0.98],
                 'adstock_7':['adstock_7',0.99], 'adstock_8':['adstock_8',0.99]}




#
#stock_details = []
#
#adstock_lambda = pd.DataFrame(stock_details)
adstock_dict = {}



def urlify(s):
    s = re.sub(r"[^\w\s]", '', s)
    s = re.sub(r"\s+", '_', s)
    return s

def lag(df, n):
    new_columns = ["{}_lag{:02d}".format(variable, n) for variable in df.columns]
    new_df = df.shift(n)
    new_df.columns = new_columns
    return new_df

def lagged_dataframe(df, lags=1):
    data_frames = [df]
    data_frames.extend([lag(df, i) for i in range(1, lags + 1)])
    return pd.concat(data_frames, axis=1)




def diffdata(df):
    n = len(df)
    newcolumns = ["{}_diff01".format(variable) for variable in df.columns]
    newdf=pd.DataFrame(np.diff(df, axis = 0))
    kdata = newdf.append(pd.Series(), ignore_index = True)
    kdata["new"]=range(1, len(kdata)+1)
    kdata.ix[n, 'new']=0
    kdata = kdata.sort_values('new').reset_index(drop='True')
    kdata.drop("new", axis = 1, inplace = True)
    kdata.columns=newcolumns
    return kdata


def diff_lag(df,k):
    dt=diffdata(df)
    finaldata=lagged_dataframe(dt,lags=k)
    return finaldata

def lag_diff(df,k):
    dt=lagged_dataframe(df, lags=k)
    finaldata=diffdata(dt)
    return finaldata


def completedata(df,k):
    n= len(df)
    ldata=lagged_dataframe(df, lags=k)
    newcolumns = ["{}_diff01".format(variable) for variable in df.columns]
    newdf=pd.DataFrame(np.diff(df, axis = 0))
    kdata = newdf.append(pd.Series(), ignore_index = True)
    kdata["new"]=range(1, len(kdata)+1)
    kdata.ix[n, 'new']=0
    kdata = kdata.sort_values('new').reset_index(drop='True')
    kdata.drop("new", axis = 1, inplace = True)
    kdata.columns=newcolumns
    #ldata=lagged_dataframe(df, lags=k)
    data_frame= ldata.join(kdata)
    return data_frame


def unusual_columns(df):
    df.fillna(0, inplace= True)
    cols=list(df.columns)
    new_cols = list(map(urlify, cols))
    new_cols = [x.lower() for x in new_cols]
    df.columns=new_cols
    type = pd.DataFrame(df.dtypes)
    type['col_names'] = df.columns
    unusual_cols =[]
    for i in range(len(type)):
        if (type.iloc[i,0]!='float64' and type.iloc[i,0]!='int64'):
            unusual_cols.append(type.iloc[i,1])        
    
    return unusual_cols


def adstock_dataframe(df, adstock_lambda,adstock_prefix_conditions, adstock_suffix_conditions):
    df.fillna(0, inplace= True)
    cols=list(df.columns)
    new_cols = list(map(urlify, cols))
    new_cols = [x.lower() for x in new_cols]
    df.columns=new_cols
    names_cols = list(df.columns)
    adstock_dict = {}
    unusual_cols = unusual_columns(df)
    for i in unusual_cols:
        names_cols.remove(i)
    adstock_1 = []
    for i in names_cols:
        for j in adstock_prefix_conditions:
            if (j in i):
                adstock_1.append(i)
    adstock_final_cols = []
    
    if len(adstock_suffix_conditions) != 0 :
        for i in adstock_1:
            for j in adstock_suffix_conditions:
                if (j in i):
                    adstock_final_cols.append(i)
    else:
        adstock_final_cols = adstock_1 
    
    adstock_df= df[adstock_final_cols]
    
    #addataframe = pd.DataFrame(columns = adstock_final_cols)
    
    for j in range(len(adstock_lambda.columns)):
        keyn = str(adstock_lambda.iloc[0,j])
        new_columns = list(adstock_df.columns)
        addataframe = pd.DataFrame(columns = new_columns)
        for i in range(len(adstock_df)):
            if i == 0:
                val= adstock_df.iloc[i,:]
                addataframe = addataframe.append(val, ignore_index = True)
                print('yes')
            else:
                val = (val*float(adstock_lambda.iloc[1,j])) + adstock_df.iloc[i,:]
                addataframe = addataframe.append(val, ignore_index = True)
        
        new_columns_up = ["{}_{:s}".format(variable, str(adstock_lambda.iloc[0,j])) for variable in adstock_df.columns]
        addataframe.columns = new_columns_up
        addataframe['tmp'] = [_ for _ in range(len(addataframe))]
        adstock_dict[keyn] = addataframe
    
    key_names = list(adstock_dict.keys())
    row = [i for i in range(len(df))]
    final_adstock_df = pd.DataFrame()
    final_adstock_df['tmp'] = row
    
    for i in key_names:
        data=adstock_dict[i]
        final_adstock_df = pd.merge(final_adstock_df, data, on = ['tmp'])
        
    return final_adstock_df

def logit(df):
    dfa = df.copy()
    dfaa =np.power((sum(np.square(np.array(dfa)))/(len(dfa)-1)), 0.5)
    for i in range(len(dfaa)):
        dfa.iloc[:,i] = dfa.iloc[:,i]/dfaa[i]
    dfa = ssp.expit(dfa)
    return dfa
    

def merged_data(df,n):
    if str(n) == 'daily':
        stock_details = daily
        lg = 30
        print(stock_details)
        
    elif str(n) == 'weekly':
        stock_details = weekly
        lg = 8
        print(stock_details)
        
    elif str(n) == 'monthly':
        lg = 2
        stock_details = monthly
        print(stock_details)
        
    adstock_lambda = pd.DataFrame(stock_details)
    unusual_col_names=unusual_columns(df)
    columns_for_lag_diff = [i for i in list(df.columns) if i not in unusual_col_names]
    df_diff_lag = df[columns_for_lag_diff]
    diff_lag_data = completedata(df_diff_lag, lg)
    diff_lag_data['tmp'] = [_i for _i in range(len(df))]
    adstock_data = adstock_dataframe(df,adstock_lambda ,adstock_prefix_conditions, adstock_suffix_conditions)
    final_master_data = pd.merge(adstock_data, diff_lag_data, on = ['tmp'])
    final_master_data.fillna(0, inplace= True)
    final_master_data.drop(columns = ['tmp'], inplace = True)
    final_master_data.to_csv('final_actual_data.csv')
    final_actual = final_master_data.copy()
    final_logit_data = logit(final_actual)
    final_logit_data.to_csv('final_logit_data.csv')
    return final_logit_data, final_master_data



#df_l, df_a = merged_data(df, 'monthly')

def summary_stats(df, tg):
    if tg == 'daily':
        n = 365
    elif tg == 'weekly':
        n = 52
    elif tg == 'monthly':
        n = 12
    df.columns = df.columns.str.lower()
    date_columns = [i for i in df.columns if 'date' in i]
    date_columns = df[date_columns]       
    df,dfa = merged_data(df,tg)    
    l2 = np.square(dfa)
    l2 = pd.DataFrame(l2.sum())
    l2.columns = ['k_value']
    l2['k_value'] = l2['k_value']/(len(dfa)-1)
    l2['k_value'] = np.power(l2['k_value'], 0.5)
    a = pd.DataFrame(df.max())
    a.columns = ['max']
    b = pd.DataFrame(df.min())
    b.columns=['min']
    c = pd.DataFrame(df.mean())
    c.columns= ['overall_average']
    d = pd.DataFrame(df.sum())
    d.columns= ['total_sum']
    d1 = pd.DataFrame(dfa.sum())
    d1.columns= ['total_sum_absolute']
    e = pd.DataFrame(dfa.astype(bool).sum(axis=0))
    e.columns = ['non_zero']
    f = pd.concat([d,e], axis = 1)
    f1 = pd.DataFrame()
    f1['active_average'] = (f['total_sum']-(0.5*(len(df)- f['non_zero'])))/f['non_zero']
    
    df_ap_last_year = df.tail(n)
    df_app_last_year = dfa.tail(n)    
    g = pd.DataFrame(df_ap_last_year.max())
    g.columns = ['max_last_year']
    h = pd.DataFrame(df_ap_last_year.min())
    h.columns=['min_last_year']
    i = pd.DataFrame(df_ap_last_year.mean())
    i.columns= ['overall_average_last_year']
    j = pd.DataFrame(df_ap_last_year.sum())
    j.columns= ['total_sum_last_year']
    j1 = pd.DataFrame(df_app_last_year.sum())
    j1.columns= ['total_sum_last_year_absolute']
    k = pd.DataFrame(df_app_last_year.astype(bool).sum(axis=0))
    k.columns = ['non_zero_last_year']
    l = pd.concat([j,k], axis = 1)
    l1 = pd.DataFrame()
    l1['active_average_last_year'] = (l['total_sum_last_year']-(0.5*(len(df_ap_last_year)- l['non_zero_last_year'])))/l['non_zero_last_year']
    summary= pd.concat([b,c,f1,i,l1,e,g,a,h,j,j1,k,d,d1,l2], axis = 1)
    summary.loc['Intercept'] = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    summary['var_names'] = [str(i) for i in summary.index]
    
    
    summary.to_csv('summary.csv')
    dfa = pd.concat([dfa, date_columns], axis = 1)
    df['null'] = 0.5
    return summary ,df,dfa



#summary, dfl, dfa = summary_stats(df,'weekly')
#dfl['null'] = 0.5

#dfl.to_csv('dfl.csv')


#summary.loc['_smr_secondary_sales_value_lac', 'k_value']
#x = pd.DataFrame(list(summary.columns))
#x['index_val'] = [i for i in range(len(summary.columns))]
#x



#formula = "kpi_all_leads ~ paid_offline_tv_grps_from_campaign_analysis + paid_offline_print_spends_grand_total_"
#
#result = sm.ols(formula, data=dfl).fit()
#
#
#par_dff = pd.DataFrame(result.params)
#
#par_val =  pd.DataFrame(result.pvalues)
#
#dependent_variable = 'kpi_all_leads'

def indx(par_df):
    list_con = ['_lag01', '_lag04', '_lag02', '_lag03', '_diff01', '_adstock01', '_adstock1', '_adstock02', '_adstock2', '_adstock03', '_adstock3', '_adstock04', '_adstock4', '_adstock05', '_adstock5', '_adstock06', '_adstock6', '_adstock07', '_adstock7', '_adstock08', '_adstock8', '_ad1', '_ad2', '_ad3', '_ad4', '_ad5', '_ad6', '_ad7','_ad8', '_adstock_1', '_adstock_2',
          '_adstock_3', '_adstock_4', '_adstock_5', '_adstock_6', '_adstock_7', '_adstock_8']

    new_idx = []
    for i in range(len(par_df)):
        k = str(par_df.index[i])
        print(k)
        for j in list_con:
            if j in k:
                k = k[:(-1*len(j))]
                new_idx.append(k)            
    old_idx = [par_df.index[l] for l in range(len(par_df))]
    for j in range(len(old_idx)):
        for i in new_idx:
            if i in (old_idx[j]):
                old_idx[j] = i
    return old_idx
    


def contribution(par_df, summ_df ,dependent_variable, p_val, custom_index, base_form):    
    k_il =[str(i) for i in par_df.index]
    old_idx = indx(par_df)
    par_df['new_idx'] = old_idx
    par_df.set_index('new_idx', inplace = True)    
    summary = summ_df
    kval = summary.loc[str(dependent_variable), 'k_value']
    print("kval is --------",kval)
    par_df['logit_min'] = [summary.loc[i,'min'] for i in (par_df.index) ]
    par_df['logit_overall_average'] = [summary.loc[i,'overall_average'] for i in (par_df.index) ]
    par_df['logit_active_average'] = [summary.loc[i,'active_average'] for i in (par_df.index) ]
    par_df['logit_overall_average_last_year'] = [summary.loc[i,'overall_average_last_year'] for i in (par_df.index) ]
    par_df['logit_active_average_last_year'] = [summary.loc[i,'active_average_last_year'] for i in (par_df.index) ]


    
    x=0
    
    for i in range(len(par_df)):
        j = custom_index[i]
        x += par_df.iloc[i,0]*par_df.iloc[i,j]

    
    par_df['predicted_value'] = x
    predicted_oa = par_df['predicted_value']    

    predicted_condition = []
    for i in range(len(par_df)):
        k=custom_index[i]
        val = par_df.iloc[i,0]*par_df.iloc[i,1]
        val1 = par_df.iloc[i,0]*par_df.iloc[i,k]
        val2 = sum(par_df.iloc[:,0]*par_df.iloc[:,2])
        val2 = val2 - val1 + val
        predicted_condition.append(val2)
    val3 = sum(par_df.iloc[:,0]*par_df.iloc[:,1])
    predicted_condition[0] = val3
    
    par_df['predicted_condition'] = predicted_condition
    
    predicted_oa_actual = []
    for i in range(len(par_df)):
        v = np.log(predicted_oa[i]/(1-predicted_oa[i]))*kval
        predicted_oa_actual.append(v)
    
    par_df['predicted_oa_actual']= predicted_oa_actual
    
    predicted_condition_actual = []
    for i in range(len(par_df)):
        v = np.log(predicted_condition[i]/(1-predicted_condition[i]))*kval
        predicted_condition_actual.append(v)
    par_df['predicted_condition_actual']= predicted_condition_actual
        
    percentage_unormalized = []
    for i in range(len(par_df)):              
        val = (par_df.iloc[i,8]-par_df.iloc[i,9])/par_df.iloc[i,9]
        percentage_unormalized.append(val)
        
    par_df['percentage_unormalized'] = percentage_unormalized
    par_df['percentage_normalized'] =percentage_unormalized/np.sum(percentage_unormalized)
    
    if base_form == 1:
        percentage_unormalized[0] = (par_df.iloc[0,8]-par_df.iloc[0,9])/par_df.iloc[0,9]
    elif base_form == 2:
        percentage_unormalized[0] = (1-(par_df.iloc[0,8]-par_df.iloc[0,9])/par_df.iloc[0,9])
    elif base_form == 3:
        percentage_unormalized[0] = (par_df.iloc[0,8]-par_df.iloc[0,9])/par_df.iloc[0,8]
    
    par_df['percentage_normalized'] =(percentage_unormalized/np.sum(percentage_unormalized))*100
    old_idxx = indx(p_val)
    p_val['new_idx'] = old_idxx
    p_val.set_index('new_idx', inplace = True)
    par_df = pd.concat([par_df, p_val], axis = 1)
        
    par_df['variables'] = [str(i) for i in par_df.index]
    
    k = []
    idk = 0
    while idk<len(par_df):
        vv = int(custom_index[idk])
        vaa = par_df.iloc[idk, int(vv)]
        k.append(vaa)
        idk = idk+1
    par_df['logit_average_used'] = k
    par_df.to_csv('cal_file.csv')
    par_d = par_df.iloc[:,[-2, 0, -4, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    par_d.columns = ['Variable', 'Coeff.' , 'percentage_normalized', 'logit_average_used', 'logit_min', 'logit_overall_average', 'logit_active_average', 'logit_overall_average_last_year', 'logit_active_average_last_year', 'predicted_value', 'predicted_condition', 'predicted_oa_actual', 'predicted_condition_actual', 'percentage_unormalized']
    par_dff = par_df.iloc[:,[-2, 0, -3, -4]]
    par_dff = pd.DataFrame(par_dff)
    par_dff.columns = ['var_names', 'estimates', 'p-values', 'contribution']
    par_d['untransformed_columns'] = k_il
    #del par_dff.index.name
    #del par_d.index.name     
    return par_dff, par_d

    
    

#custom_idx = [2,2,2,2,2,2]

#base_form = 3

#parad = contribution(par_dff,summary ,dependent_variable, par_val, custom_idx , base_form)

#parad.to_csv('parad.csv')

#parad['var_name'] = [str(i) for i in parad.index]

#par_dff
    