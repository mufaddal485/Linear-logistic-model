
# coding: utf-8

# In[190]:


import numpy as np
import pandas as pd


# In[191]:


data_file='D:\Mufaddal\Data Science\Python\Data Sets\loans data.csv'
ld=pd.read_csv(data_file)


# In[192]:


ld.head()


# In[193]:


ld.dtypes


# In[194]:


for col in ['Interest.Rate','Debt.To.Income.Ratio']:
    ld[col]=ld[col].astype('str')
    ld[col]=[x.replace('%','') for x in ld[col]]


# In[195]:


for col in ["Amount.Requested","Amount.Funded.By.Investors","Open.CREDIT.Lines","Revolving.CREDIT.Balance",
           "Inquiries.in.the.Last.6.Months","Interest.Rate","Debt.To.Income.Ratio",]:
    ld[col]=pd.to_numeric(ld[col],errors='coerce')


# In[196]:


ld['Loan.Length'].value_counts()


# In[197]:


ll_dummies=pd.get_dummies(ld["Loan.Length"])


# In[198]:


ll_dummies.head()


# In[199]:


ld["LL_36"]=ll_dummies["36 months"]


# In[200]:


get_ipython().run_line_magic('reset_selective', 'll_dummies')


# In[201]:


who


# In[202]:


ld.dtypes


# In[203]:


ld=ld.drop('Loan.Length',axis=1)


# In[204]:


round(ld.groupby("Loan.Purpose")["Interest.Rate"].mean())


# In[205]:


for i in range(len(ld.index)):
    if ld["Loan.Purpose"][i] in ["car","educational","major_purchase"]:
        ld.loc[i,"Loan.Purpose"]="cem"
    if ld["Loan.Purpose"][i] in ["home_improvement","medical","vacation","wedding"]:
        ld.loc[i,"Loan.Purpose"]="hmvw"
    if ld["Loan.Purpose"][i] in ["credit_card","house","other","small_business"]:
        ld.loc[i,"Loan.Purpose"]="chos"
    if ld["Loan.Purpose"][i] in ["debt_consolidation","moving"]:
        ld.loc[i,"Loan.Purpose"]="dm"


# In[206]:


lp_dummies=pd.get_dummies(ld["Loan.Purpose"],prefix="LP")


# In[207]:


lp_dummies.head()


# In[208]:


ld=pd.concat([ld,lp_dummies],1)
ld=ld.drop(["Loan.Purpose","LP_renewable_energy"],1)


# In[209]:


ld.dtypes


# In[210]:


ld=ld.drop(["State"],1)


# In[211]:


ld["ho_mort"]=np.where(ld["Home.Ownership"]=="MORTGAGE",1,0)
ld["ho_rent"]=np.where(ld["Home.Ownership"]=="RENT",1,0)
ld=ld.drop(["Home.Ownership"],1)


# In[212]:


ld['f1'], ld['f2'] = zip(*ld['FICO.Range'].apply(lambda x: x.split('-', 1)))


# In[213]:


ld["fico"]=0.5*(pd.to_numeric(ld["f1"])+pd.to_numeric(ld["f2"]))

ld=ld.drop(["FICO.Range","f1","f2"],1)


# In[214]:


ld.dtypes


# In[215]:


ld["Employment.Length"]=ld["Employment.Length"].astype("str")
ld["Employment.Length"]=[x.replace("years","") for x in ld["Employment.Length"]]
ld["Employment.Length"]=[x.replace("year","") for x in ld["Employment.Length"]]


# In[216]:


ld["Employment.Length"].value_counts()


# In[217]:


ld["Employment.Length"]=[x.replace("n/a","< 1") for x in ld["Employment.Length"]]
ld["Employment.Length"]=[x.replace("10+","10") for x in ld["Employment.Length"]]
ld["Employment.Length"]=[x.replace("< 1","0") for x in ld["Employment.Length"]]
ld["Employment.Length"]=pd.to_numeric(ld["Employment.Length"],errors="coerce")


# In[218]:


ld.dtypes


# In[219]:


ld.shape


# In[220]:


ld.dropna(axis=0,inplace=True)


# In[221]:



import math
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.cross_validation import KFold
get_ipython().run_line_magic('matplotlib', 'inline')


# In[222]:


ld_train, ld_test = train_test_split(ld, test_size = 0.2,random_state=2)


# In[223]:


lm=LinearRegression()


# In[224]:


x_train=ld_train.drop(["Interest.Rate","ID","Amount.Funded.By.Investors"],1)
y_train=ld_train["Interest.Rate"]
x_test=ld_test.drop(["Interest.Rate","ID","Amount.Funded.By.Investors"],1)
y_test=ld_test["Interest.Rate"]


# In[225]:


lm.fit(x_train,y_train)


# In[226]:


p_test=lm.predict(x_test)

residual=p_test-y_test

rmse_lm=np.sqrt(np.dot(residual,residual)/len(p_test))

rmse_lm


# In[227]:


coefs=lm.coef_

features=x_train.columns

list(zip(features,coefs))


# In[228]:


alphas=np.linspace(.0001,10,100)
x_train.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)


# In[229]:


rmse_list=[]
for a in alphas:
    ridge = Ridge(fit_intercept=True, alpha=a)

    # computing average RMSE across 10-fold cross validation
    kf = KFold(len(x_train), n_folds=10)
    xval_err = 0
    for train, test in kf:
        ridge.fit(x_train.loc[train], y_train[train])
        p = ridge.predict(x_train.loc[test])
        err = p - y_train[test]
        xval_err += np.dot(err,err)
    rmse_10cv = np.sqrt(xval_err/len(x_train))
    # uncomment below to print rmse values for individidual alphas
#     print('{:.3f}\t {:.6f}\t '.format(a,rmse_10cv))
    rmse_list.extend([rmse_10cv])
best_alpha=alphas[rmse_list==min(rmse_list)]
print('Alpha with min 10cv error is : ',best_alpha )


# In[230]:



ridge=Ridge(fit_intercept=True,alpha=best_alpha)

ridge.fit(x_train,y_train)

p_test=ridge.predict(x_test)

residual=p_test-y_test

rmse_ridge=np.sqrt(np.dot(residual,residual)/len(p_test))

rmse_ridge


# In[231]:


alphas=np.linspace(0.0001,1,100)
rmse_list=[]
for a in alphas:
    lasso = Lasso(fit_intercept=True, alpha=a,max_iter=10000)

    # computing RMSE using 10-fold cross validation
    kf = KFold(len(x_train), n_folds=10)
    xval_err = 0
    for train, test in kf:
        lasso.fit(x_train.loc[train], y_train[train])
        p =lasso.predict(x_train.loc[test])
        err = p - y_train[test]
        xval_err += np.dot(err,err)
    rmse_10cv = np.sqrt(xval_err/len(x_train))
    rmse_list.extend([rmse_10cv])
    # Uncomment below to print rmse values of individual alphas
   # print('{:.3f}\t {:.4f}\t '.format(a,rmse_10cv))
best_alpha=alphas[rmse_list==min(rmse_list)]
print('Alpha with min 10cv error is : ',best_alpha )


# In[232]:


lasso=Lasso(fit_intercept=True,alpha=best_alpha)

lasso.fit(x_train,y_train)

p_test=lasso.predict(x_test)

residual=p_test-y_test

rmse_lasso=np.sqrt(np.dot(residual,residual)/len(p_test))

rmse_lasso


# In[233]:


list(zip(x_train.columns,lasso.coef_))


# In[234]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# In[235]:


file_name='D:\Mufaddal\Data Science\Python\Data Sets\Existing Base.csv'
bd=pd.read_csv(file_name)


# In[236]:


bd.head()


# In[237]:


bd.loc[bd["children"]=="Zero","children"]="0"
bd.loc[bd["children"]=="4+","children"]="4"
bd["children"]=pd.to_numeric(bd["children"],errors="coerce")


# In[238]:


bd["Revenue Grid"].value_counts()


# In[239]:


bd["y"]=np.where(bd["Revenue Grid"]==2,0,1)
bd=bd.drop(["Revenue Grid"],1)


# In[240]:


round(bd.groupby("age_band")["y"].mean(),2)


# In[241]:


for i in range(len(bd)):
    if bd["age_band"][i] in ["71+","65-70","51-55","45-50"]:
        bd.loc[i,"age_band"]="ab_10"
    if bd["age_band"][i] in ["55-60","41-45","31-35","22-25","26-30"]:
        bd.loc[i,"age_band"]="ab_11"
    if bd["age_band"][i]=="36-40":
        bd.loc[i,"age_band"]="ab_13"
    if bd["age_band"][i]=="18-21":
        bd.loc[i,"age_band"]="ab_17"
    if bd["age_band"][i]=="61-65":
        bd.loc[i,"age_band"]="ab_9"
ab_dummies=pd.get_dummies(bd["age_band"])
ab_dummies.head()


# In[242]:


bd=pd.concat([bd,ab_dummies],1)
bd=bd.drop(["age_band","Unknown"],1)


# In[243]:


bd["st_partner"]=np.where(bd["status"]=="Partner",1,0)
bd["st_singleNm"]=np.where(bd["status"]=="Single/Never Married",1,0)
bd["st_divSep"]=np.where(bd["status"]=="Divorced/Separated",1,0)
bd=bd.drop(["status"],1)


# In[244]:


for i in range(len(bd)):
    if bd["occupation"][i] in ["Unknown","Student","Secretarial/Admin","Other","Manual Worker"]:
        bd.loc[i,"occupation"]="oc_11"
    if bd["occupation"][i] in ["Professional","Business Manager"]:
        bd.loc[i,"occupation"]="oc_12"
    if bd["occupation"][i]=="Retired":
        bd.loc[i,"occupation"]="oc_10"
oc_dummies=pd.get_dummies(bd["occupation"])
oc_dummies.head()


# In[245]:


bd=pd.concat([bd,oc_dummies],1)

bd=bd.drop(["occupation","Housewife"],1)


# In[246]:


round(bd.groupby("occupation_partner")["y"].mean(),2)


# In[247]:


bd["ocp_10"]=0
bd["ocp_12"]=0
for i in range(len(bd)):
    if bd["occupation_partner"][i] in ["Unknown","Retired","Other"]:
        bd.loc[i,"ocp_10"]=1
    if bd["occupation_partner"][i] in ["Student","Secretarial/Admin"]:
        bd.loc[i,"ocp_12"]=1
        
bd=bd.drop(["occupation_partner","TVarea","post_code","post_area","region"],1)


# In[248]:


bd["home_status"].value_counts()


# In[249]:


bd["hs_own"]=np.where(bd["home_status"]=="Own Home",1,0)
del bd["home_status"]


# In[250]:


bd["gender_f"]=np.where(bd["gender"]=="Female",1,0)
del bd["gender"]


# In[251]:


bd["semp_yes"]=np.where(bd["self_employed"]=="Yes",1,0)
del bd["self_employed"]


# In[252]:


bd["semp_part_yes"]=np.where(bd["self_employed_partner"]=="Yes",1,0)
del bd["self_employed_partner"]


# In[253]:


round(bd.groupby("family_income")["y"].mean(),4)


# In[254]:


bd["fi"]=4 # by doing this , we have essentially clubbed <4000 and Unknown values . How?
bd.loc[bd["family_income"]=="< 8,000, >= 4,000","fi"]=6
bd.loc[bd["family_income"]=="<10,000, >= 8,000","fi"]=9
bd.loc[bd["family_income"]=="<12,500, >=10,000","fi"]=11.25
bd.loc[bd["family_income"]=="<15,000, >=12,500","fi"]=13.75
bd.loc[bd["family_income"]=="<17,500, >=15,000","fi"]=16.25
bd.loc[bd["family_income"]=="<20,000, >=17,500","fi"]=18.75
bd.loc[bd["family_income"]=="<22,500, >=20,000","fi"]=21.25
bd.loc[bd["family_income"]=="<25,000, >=22,500","fi"]=23.75
bd.loc[bd["family_income"]=="<27,500, >=25,000","fi"]=26.25
bd.loc[bd["family_income"]=="<30,000, >=27,500","fi"]=28.75
bd.loc[bd["family_income"]==">=35,000","fi"]=35
bd=bd.drop(["family_income"],1)


# In[255]:


bd.dtypes


# In[256]:


bd.dropna(axis=0,inplace=True)
bd_train, bd_test = train_test_split(bd, test_size = 0.2,random_state=2)


# In[257]:


x_train=bd_train.drop(["y","REF_NO"],1)
y_train=bd_train["y"]
x_test=bd_test.drop(["y","REF_NO"],1)
y_test=bd_test["y"]


# In[258]:


logr=LogisticRegression(penalty="l1",class_weight="balanced",random_state=2)


# In[259]:


logr.fit(x_train,y_train)


# In[260]:


# score model performance on the test data
roc_auc_score(y_test,logr.predict(x_test))


# In[261]:


prob_score=pd.Series(list(zip(*logr.predict_proba(x_train)))[1])


# In[262]:


cutoffs=np.linspace(0,1,100)


# In[263]:


KS_cut=[]
for cutoff in cutoffs:
    predicted=pd.Series([0]*len(y_train))
    predicted[prob_score>cutoff]=1
    df=pd.DataFrame(list(zip(y_train,predicted)),columns=["real","predicted"])
    TP=len(df[(df["real"]==1) &(df["predicted"]==1) ])
    FP=len(df[(df["real"]==0) &(df["predicted"]==1) ])
    TN=len(df[(df["real"]==0) &(df["predicted"]==0) ])
    FN=len(df[(df["real"]==1) &(df["predicted"]==0) ])
    P=TP+FN
    N=TN+FP
    KS=(TP/P)-(FP/N)
    KS_cut.append(KS)

cutoff_data=pd.DataFrame(list(zip(cutoffs,KS_cut)),columns=["cutoff","KS"])

KS_cutoff=cutoff_data[cutoff_data["KS"]==cutoff_data["KS"].max()]["cutoff"]


# In[264]:


# Performance on test data
prob_score_test=pd.Series(list(zip(*logr.predict_proba(x_test)))[1])

predicted_test=pd.Series([0]*len(y_test))
predicted_test[prob_score_test>float(KS_cutoff)]=1

df_test=pd.DataFrame(list(zip(y_test,predicted_test)),columns=["real","predicted"])

k=pd.crosstab(df_test['real'],df_test["predicted"])
print('confusion matrix :\n \n ',k)
TN=k.iloc[0,0]
TP=k.iloc[1,1]
FP=k.iloc[0,1]
FN=k.iloc[1,0]
P=TP+FN
N=TN+FP


# In[265]:


# Accuracy of test
(TP+TN)/(P+N)


# In[266]:


# Sensitivity on test
TP/P


# In[267]:


#Specificity on test
TN/N


# In[268]:


alphas=np.linspace(.0001,10,100)
x_train.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)

