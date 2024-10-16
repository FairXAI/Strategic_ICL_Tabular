
from utils import sampleRatio
import pandas as pd

# follow the data cleaning preprocess of https://arxiv.org/pdf/2310.14607.pdf ,omit Education-Num and Fnlwgt as they
# are not crucial for income prediction,
def loadCredit():
  df = pd.read_csv("ds/credit/data.csv")
  df.keys()
  # EDUCATION
  df['EDUCATION'].replace({
    1: 'graduate school',
    2: 'university',
    3: 'high school',
    4: 'others',
    5: 'unknown',
    6: 'unknown'
  }, inplace=True)

  # MARRIAGE
  df['MARRIAGE'].replace({
    1: 'married',
    2: 'single',
    3: 'others'
  }, inplace=True)

  # PAY_0 to PAY_6
  columns_to_replace = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
  replacement_dict = {
    -2: 'delay for two months',
    -1: 'pay duly',
    0: 'no consumption',
    1: 'delay for one month',
    2: 'delay for two months',
    3: 'delay for three months',
    4: 'delay for four months',
    5: 'delay for five months',
    6: 'delay for six months',
    7: 'delay for seven months',
    8: 'delay for eight months',
    9: 'delay for nine months and above'
  }
  df['SEX'].replace({1: 'male', 2: 'female'}, inplace=True)
  df.rename(columns={'LIMIT_BAL': 'Amount of given credit'}, inplace=True)
  df.rename(columns={'default payment next month': 'default payment'}, inplace=True)
  for column in columns_to_replace:
    df[column].replace(replacement_dict, inplace=True)
  df['default payment'].replace({1: 'yes,overdue.', 0: 'no,on-time.'}, inplace=True)
  bill_amt_columns = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
  pay_amt_columns = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
  df['AVG_BILL_AMT'] = df[bill_amt_columns].mean(axis=1).astype(int)
  df['AVG_PAY_AMT'] = df[pay_amt_columns].mean(axis=1).astype(int)

  df.drop(columns=bill_amt_columns + pay_amt_columns, inplace=True)
  df.drop(columns='ID', inplace=True)
  df['default payment'] = df.pop('default payment')
  return df

def readAndProcess(path, column_names, drop_names):
  df = pd.read_csv(path, names=column_names, skipinitialspace=True)
  df = df.drop(drop_names, axis=1)
  return df


def loadAdult():
    tr_path = 'ds/adult/adult.data'
    test_path = 'ds/adult/adult.test'
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                    'income']
    drop_names = ['education-num', 'fnlwgt', 'race', 'native-country']

    dftr = readAndProcess(tr_path, column_names, drop_names)
    dftst = readAndProcess(test_path, column_names, drop_names)
    return dftr,dftst

def convertAdultIncome(dfin):
  df=dfin.copy(deep=True)
  df.rename(columns={'income': 'income answer'}, inplace=True)
  df['income answer'] = df['income answer'].replace('<=50K', 'less than or equal to 50K')
  df['income answer'] = df['income answer'].replace('>50K', 'greater than 50K')
  return df

def getCreditKshot(dftr,k,ch):
  seed = 42
  num_tr = k
  ratio = 0.5#balanced
  y_name = 'default payment'
  y_name_set = ['yes,overdue.', 'no,on-time.']
  zname='SEX'
  z_name_set = ['female', 'male']
  if ch=='Balanced_FM_OntimDue':
    col_name = [zname, y_name]
    name_list = [z_name_set, y_name_set]
    sdftr, xtr, ytr = sampleRatio(dftr, col_name, name_list, num_tr, ratio, seed, y_name)
  elif ch=='All_F_Balanced_OntimDue':
    df_female = dftr[dftr[zname]=='female']
    col_name = [y_name]
    name_list = [y_name_set]
    sdftr, xtr, ytr = sampleRatio(df_female, col_name, name_list, num_tr, ratio, seed, y_name)
  elif ch=='All_F_All_Ontime':
    df_female = dftr[dftr[zname] == 'female']
    sdftr = df_female[df_female[y_name] == 'no,on-time.'].sample(n=num_tr, random_state=seed)
    xtr=sdftr.copy(deep=True).drop(y_name, axis=1)
  elif ch=='All_F_All_Overdue':
    df_female = dftr[dftr[zname] == 'female']
    sdftr = df_female[df_female[y_name] == 'yes,overdue.'].sample(n=num_tr, random_state=seed)
    xtr=sdftr.copy(deep=True).drop(y_name, axis=1)
  elif ch=='All_M_All_Overdue':
    df_female = dftr[dftr[zname] == 'male']
    sdftr = df_female[df_female[y_name] == 'yes,overdue.'].sample(n=num_tr, random_state=seed)
    xtr=sdftr.copy(deep=True).drop(y_name, axis=1)

  return sdftr, xtr, sdftr[y_name]


def getAdultKshot(dftr,k,ch):
  seed = 42
  num_tr = k
  ratio = 0.5
  y_name = 'income'

  if ch=='Balanced_FM_LH':
    col_name = ['sex', 'income']
    name_list = [['Female', 'Male'], ['<=50K', '>50K']]
    sdftr, xtr, ytr = sampleRatio(dftr, col_name, name_list, num_tr, ratio, seed, y_name)
  elif ch=='All_F_Balanced_LH':
    df_female = dftr[dftr['sex'] == 'Female']
    col_name = ['income']
    name_list = [['<=50K', '>50K']]
    sdftr, xtr, ytr = sampleRatio(df_female, col_name, name_list, num_tr, ratio, seed, y_name)
  elif ch=='All_F_All_L':
    df_female = dftr[dftr['sex'] == 'Female']
    sdftr = df_female[df_female['income'] == '<=50K'].sample(n=num_tr, random_state=seed)
    xtr=sdftr.copy(deep=True).drop(y_name, axis=1)
  elif ch=='All_F_All_H':
    df_female = dftr[dftr['sex'] == 'Female']
    sdftr = df_female[df_female['income'] == '>50K'].sample(n=num_tr, random_state=seed)
    xtr=sdftr.copy(deep=True).drop(y_name, axis=1)
  elif ch=='All_M_All_H':
    df_female = dftr[dftr['sex'] == 'Male']
    sdftr = df_female[df_female['income'] == '>50K'].sample(n=num_tr, random_state=seed)
    xtr=sdftr.copy(deep=True).drop(y_name, axis=1)

  sdftr_after = convertAdultIncome(sdftr)

  return sdftr_after, xtr, sdftr_after['income answer']

def getAdultTst(dftst,args):
  col_name_tst = ['sex', 'income']
  name_list_tst = [['Male', 'Female'], ['<=50K.', '>50K.']]
  num_tst = 512
  ratio_tst = 0.5
  seed = args.seed
  y_name = 'income'
  sdftst, xtst, ytst = sampleRatio(dftst, col_name_tst, name_list_tst, num_tst, ratio_tst, seed, y_name)  # testing
  return sdftst, xtst, ytst


def getCreditTst(dftst,args):
  y_name = 'default payment'
  y_name_set = ['yes,overdue.', 'no,on-time.']
  col_name_tst = ['SEX', y_name]
  name_list_tst = [['female', 'male'], y_name_set]
  num_tst = 512
  ratio_tst = 0.5
  seed=args.seed
  sdftst, xtst, ytst = sampleRatio(dftst, col_name_tst, name_list_tst, num_tst, ratio_tst, seed, y_name)
  return sdftst, xtst, ytst

