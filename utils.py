
import pandas as pd
import time
import os

#few-shot -> str
def fewShotMerge2str(df, row_index):
    if row_index < 0 or row_index >= len(df):
        return "Invalid row index"
    row_data = df.iloc[row_index]# get data from ith row

    result_str = ' '.join([f'{col} is {data}, and' for col, data in row_data.items()])
    result_str=f' Example {row_index}:'+result_str

    return result_str[:-5]


def sampleFun(df, col_name, name1, name2, num1, num2, seed):
  sample1 = df[df[col_name] == name1].sample(n=num1, random_state=seed)  # major group
  sample2 = df[df[col_name] == name2].sample(n=num2, random_state=seed)  # under_represented
  sampled_df = pd.concat([sample1, sample2])
  return sampled_df


# sample with fixed ratio
def sampleRatio(df,col_name,name_list,num,ratio,seed,y_name,skip=False):
  #not sample
  if skip==True:
    return df,df.copy(deep=True).drop(y_name, axis=1),df[y_name]

  if(len(name_list)==1):
    name1=name_list[0][0]
    name2=name_list[0][1]
    num1=round(num * ratio)
    num2=num-num1
    sampled_df=sampleFun(df,col_name[0],name1,name2,num1,num2,seed)

  if(len(name_list)==2):
    name1=name_list[0][0]
    name2=name_list[0][1]
    m_subset = df[df[col_name[0]] == name1]#major group
    f_subset = df[df[col_name[0]] == name2]#under_represented

    name11=name_list[1][0]
    name22=name_list[1][1]
    num1=round(num * ratio*0.5)
    num2=round(num*0.5-num1)

    sample1=sampleFun(m_subset,col_name[1],name11,name22,num1,num2,seed)#major
    #print(sample1)
    sample2=sampleFun(f_subset,col_name[1],name11,name22,num1,num2,seed)#under_rep
    #print(sample2)
    sampled_df=pd.concat([sample1, sample2])

  X = sampled_df.copy(deep=True).drop(y_name, axis=1)
  Y=sampled_df[y_name]
  return sampled_df,X,Y

#credit dataset/testing samples
def merge2stringCredit(df, row_index):
    if row_index < 0 or row_index >= len(df):
        return "Invalid row index"
    row_data = df.iloc[row_index]
    result_str = ' '.join([f'{col} is {data}, and' for col, data in row_data.items()])
    #result_str
    return result_str+' answer credit score:'


def merge2stringAdult(df, row_index):
  if row_index < 0 or row_index >= len(df):
    return "Invalid row index"
  row_data = df.iloc[row_index]
  result_str = ' '.join([f'{col} is {data}, and' for col, data in row_data.items()])
  # result_str
  return result_str + ' answer income:'


def split_by_sens(bin_ytst, bin_ypre, sens):
    group_0_indices = [i for i, value in enumerate(sens) if value == 0]
    group_1_indices = [i for i, value in enumerate(sens) if value == 1]

    group_0_ytst = [bin_ytst[i] for i in group_0_indices]
    group_0_ypre = [bin_ypre[i] for i in group_0_indices]

    group_1_ytst = [bin_ytst[i] for i in group_1_indices]
    group_1_ypre = [bin_ypre[i] for i in group_1_indices]

    return group_0_ytst, group_0_ypre, group_1_ytst, group_1_ypre

