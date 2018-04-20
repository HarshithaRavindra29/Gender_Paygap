# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:40:56 2018

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

def PG_analysis(employee_data,GPG_threshold):

    if GPG_threshold=="":
        GPG_threshold = 3
    else:

        GPG_threshold=int(GPG_threshold)

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
    
    	
    	data = [{'jobTitle': i, 'gender_pval': gender_pval,'Perc_gender_gap':round(male_coef,3) * 100,
              'MAPE_test':MAPE_test,'MAPE_Train':MAPE_train,
              'median_male_sal':median_male_sal,
              'median_Female_sal':median_Female_sal,
              'male_Emp_count':male_count,
              'Female_Emp_count':Female_count,
              'total_emp':total_emp}]
    	model_summary = pd.DataFrame(data)
        
    	if roles[0] == i :
    		All_job_result = model_summary.copy(deep = False)
    	else:
    		All_job_result = All_job_result.append(model_summary)
    
    
    print type(GPG_threshold),GPG_threshold
    Sig =  All_job_result['gender_pval']< 0.05
    Accurate = All_job_result['MAPE_test'] < 30
    pay_gap_pos = All_job_result['Perc_gender_gap'] > GPG_threshold
    pay_gap_neg = All_job_result['Perc_gender_gap'] < -GPG_threshold
    
    Pay_gap_table = All_job_result[Sig & Accurate & (pay_gap_pos | pay_gap_neg)]
    
    Pay_gap_table['Pay_gap_comment']= 'Pay gap of '+ Pay_gap_table['Perc_gender_gap'].astype(str)+ ' percent'
    
    No_pay_gap_table = All_job_result[~ (Sig & Accurate & (pay_gap_pos | pay_gap_neg))]
    No_pay_gap_table['Pay_gap_comment']= 'No pay gap'
    
    
    Full_summary = pd.concat([Pay_gap_table, No_pay_gap_table])
    
    no_of_titles_wPG = len(Pay_gap_table)
    total_titles = len(Full_summary)
    PG_intensity = no_of_titles_wPG/total_titles
    
    if PG_intensity > 40:
        Final_comment = PG_intensity.astype(str)+"% of your department are afftected by gender pay gap, please correct it using our tool"
    else:
        Final_comment = "Congratultion! Your company practices fair salary policy"
    
    #No_Pay_gap_table_names = [No_pay_gap_table.JobTitle.drop_duplicates().values]
    Pay_gap_graph_names = Pay_gap_table.jobTitle.values.tolist()
    Pay_gap_graph_values = Pay_gap_table.Perc_gender_gap.values.tolist()
    Pay_gap_table_stats = Full_summary[['jobTitle','Pay_gap_comment','median_Female_sal','median_male_sal','male_Emp_count','Female_Emp_count']]   
    return Pay_gap_graph_names,Pay_gap_graph_values,Pay_gap_table_stats,Final_comment

#Pay_gap_graph_names,Pay_gap_graph_values,Pay_gap_table_stats,Final_comment = PG_analysis(employee_data,GPG_threshold)