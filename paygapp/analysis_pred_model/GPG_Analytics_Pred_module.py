# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 22:33:43 2018

@author: Harshitha R
"""

import os
import pandas as pd
import numpy as np
from sklearn import linear_model as lm
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

#os.chdir("C:\Users\User\Desktop\GPG\Test_data")


employee_data = pd.read_csv("Glass_Door_data.csv")
GPG_threshold = 3
"""
JT is Job title
JT='Driver'
PE is the Performance Evaluation - Interview score reduced to range of 1-5
PE = 3
SE is the seniority/experience
Se = 3
Ed is the Education
Ed = 'College'
"""
def Salary_Prediction(employee_data,GPG_threshold,JT,PE,Se,Ed):
    if GPG_threshold == "":
        GPG_threshold = 3
    else:

        GPG_threshold = int(GPG_threshold)

    def gender_code_fun(row):
        if row["gender"].lower() == "male": 
            return 1
        else: 
            return 0
    
    employee_data = employee_data.assign(gender_code=employee_data.apply(gender_code_fun, axis=1))
    
    edu_dummies = pd.get_dummies(employee_data.edu)
    employee_data = employee_data.join(edu_dummies)
    
    
    employee_data['totalSalary'] = employee_data.basePay + employee_data.bonus
    employee_data['totalSalary_log'] = np.log(employee_data['totalSalary'])
    x = employee_data.edu.drop_duplicates().values
    model_var = employee_data[['jobTitle','perfEval','seniority','totalSalary','totalSalary_log','gender_code'] + list(x)]
    
    roles = employee_data['jobTitle'].drop_duplicates().values
    
    for i in roles:
    	print i
    
    	model_sub = model_var[model_var.jobTitle == i]
    	median_male_sal = np.median(model_sub[model_sub.gender_code == 1].totalSalary)
    	median_Female_sal = np.median(model_sub[model_sub.gender_code == 0].totalSalary)
    	Female_count = len(model_sub[model_sub.gender_code == 0].index)
    	male_count = len(model_sub[model_sub.gender_code == 1].index)
    	total_emp = len(model_sub.totalSalary.index)
        
        
    	X = model_sub[['perfEval','seniority','gender_code'] + list(x)]
    	y = model_sub[['totalSalary_log']]
    	
    	X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
    	
    	reg1 = lm.LinearRegression()
    	reg1.fit(X_train,y_train)
    	y_pred = reg1.predict(X_test)
    	
    	def mean_absolute_percentage_error(y_true, y_pred): 
    		y_true, y_pred = np.array(y_true), np.array(y_pred)
    		return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    	MAPE_test = mean_absolute_percentage_error(np.exp(y_test),np.exp(y_pred))
    	MAPE_train = mean_absolute_percentage_error(np.exp(y), np.exp(reg1.predict(X)))
    	
    	
    	reg2 = sm.OLS(y,X).fit()
    	gender_pval = reg2.pvalues.gender_code
    	male_coef = reg2.params.gender_code
    
    	
    	sum_data = [{'jobTitle': i, 'gender_pval': gender_pval,'Perc_gender_gap':round(male_coef,3) * 100,
              'MAPE_test':MAPE_test,'MAPE_Train':MAPE_train,
              'median_male_sal':median_male_sal,
              'median_Female_sal':median_Female_sal,
              'male_Emp_count':male_count,
              'Female_Emp_count':Female_count,
              'total_emp':total_emp}]
    	model_summary = pd.DataFrame(sum_data)
        
    	if roles[0] == i :
    		All_job_result = model_summary.copy(deep = False)
    	else:
    		All_job_result = All_job_result.append(model_summary)
    

    #change
    Sig =  All_job_result['gender_pval']< 0.05
    Accurate = All_job_result['MAPE_test'] < 30
    pay_gap_pos = All_job_result['Perc_gender_gap'] > GPG_threshold
    pay_gap_neg = All_job_result['Perc_gender_gap'] < -GPG_threshold
    
    Pay_gap_table = All_job_result[Sig & Accurate & (pay_gap_pos | pay_gap_neg)]
    
    Pay_gap_table['Pay_gap_comment']= 'Pay gap of '+ Pay_gap_table['Perc_gender_gap'].astype(str)+ ' percent'
    
    No_pay_gap_table = All_job_result[~ (Sig & Accurate & (pay_gap_pos | pay_gap_neg))]
    No_pay_gap_table['Pay_gap_comment']= 'No pay gap'
    
    
   # Full_summary = pd.concat([Pay_gap_table, No_pay_gap_table])
    
    Sal_fine = employee_data[~employee_data['jobTitle'].isin(Pay_gap_table['jobTitle'].values)]
    
    Sal_to_adj = pd.merge(employee_data,Pay_gap_table[['jobTitle','Perc_gender_gap']],how = 'inner',on ='jobTitle')
    
    
    Sal_to_adj['Fem_adj'] = Sal_to_adj['totalSalary'] + (Sal_to_adj['totalSalary'] * Sal_to_adj['Perc_gender_gap']/100)
    Sal_to_adj['male_adj'] = Sal_to_adj['totalSalary'] - (Sal_to_adj['totalSalary'] * Sal_to_adj['Perc_gender_gap']/100)
    
    Sal_to_adj['Sal_adj_ful'] = np.where(((Sal_to_adj['gender'] == 'Female') & (Sal_to_adj['Perc_gender_gap'] > GPG_threshold)),
              Sal_to_adj['Fem_adj'] ,np.where(((Sal_to_adj['gender'] == 'Male') & (Sal_to_adj['Perc_gender_gap'] < -GPG_threshold)),Sal_to_adj['male_adj']
                        ,Sal_to_adj['totalSalary']))
    
    Sal_to_adj.totalSalary = Sal_to_adj['Sal_adj_ful']
    Sal_to_adj.totalSalary_log = np.log(Sal_to_adj.totalSalary)
    Sal_to_adj = Sal_to_adj.drop(['Perc_gender_gap', 'Fem_adj', 'male_adj','Sal_adj_ful'], axis = 1)
    
    Adj_Emp_data = pd.concat([Sal_to_adj, Sal_fine])
    
    model_var_n = Adj_Emp_data[['jobTitle','perfEval','seniority','totalSalary','totalSalary_log'] + list(x)]
    
    
    
    def Predict_NR_Sal(JT,PE,Se,Ed):
    	model_subn = model_var_n[model_var_n.jobTitle == JT]
    	median_saln = np.median(model_subn.totalSalary)
    	
    	total_empn = len(model_sub.totalSalary.index)
        
        
    	Xn = model_sub[['perfEval','seniority'] + list(x)]
    	yn = model_sub[['totalSalary_log']]
    		
    	regP = lm.LinearRegression()
    	regP.fit(Xn,yn)
    	Ed_dash = dict(zip(x[x != Ed], [0,0,0]))
    	inp_var = {'perfEval':PE,'seniority':Se,Ed:1}
    	inp_var.update(Ed_dash)
    	Candidate_data = pd.DataFrame([inp_var])
    	Sal_pred = np.exp(regP.predict(Candidate_data))
    	return(Sal_pred,median_saln,total_empn)
    Sal_pred,median_saln,total_empn = Predict_NR_Sal(JT,PE,Se,Ed)
    return(Sal_pred,median_saln,total_empn)

#a,b,c=Salary_Prediction(employee_data,GPG_threshold,JT,PE,Se,Ed)