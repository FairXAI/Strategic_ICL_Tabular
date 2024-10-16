
from dataprocess import loadAdult,loadCredit,getAdultTst,getAdultKshot,getCreditKshot,getCreditTst
from utils import merge2stringCredit,merge2stringAdult,fewShotMerge2str
from sklearn.model_selection import train_test_split
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

def pipeAdultData(args):
  # Write your customized background introduction. For example, predict if income exceeds $50K per year: greater than 50K | less than or equal to 50K.
  intro = "xxxxxxxxxxxxxxxxxxxx"
  dftr, dftst = loadAdult()
  dftr, dfdev= train_test_split(dftr, test_size=0.1, random_state=args.seed)#validation set

  sdftst, xtst, ytst = getAdultTst(dftst,args)
  sdftr, xtr, ytr = getAdultKshot(dftr, args.k, args.strategy)
  few_prompt=""
  for i in range(len(sdftr)):
    few_prompt=few_prompt+fewShotMerge2str(sdftr,i)
  return intro,few_prompt,sdftst

def pipeCreditData(args):
  # Write your customized background introduction. For example: Answer whether default payment/overdue: No | Yes
  #if GPT4 refuse to answer use followings, try sth like "Suppose that you are a ML prediction model, xxxxxxx"
  intro = "xxxxxxxxxxxxxxxxxxxx"

  df = loadCredit()
  dftr, dftst = train_test_split(df, test_size=0.8, random_state=args.seed)
  dftr, dfdev  = train_test_split(dftr, test_size=0.1, random_state=args.seed)

  sdftst, xtst, ytst = getCreditTst(dftst,args)
  sdftr, xtr, ytr = getCreditKshot(dftr, args.k, args.strategy)
  few_prompt=""
  for i in range(len(sdftr)):
    few_prompt=few_prompt+fewShotMerge2str(sdftr,i)
  return intro,few_prompt,xtst



def askGPT(args,client,intro,fewpromp,xtst):
  @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(20))
  def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

  for i in range(0, len(xtst)):
    ins_tmp=""
    if args.dataset=='Adult':
      ins_tmp = merge2stringAdult(xtst, i)
      print('i', i, ' ', ins_tmp)

    if args.dataset=='Credit':
      ins_tmp = merge2stringCredit(xtst, i)
      print('i', i, ' ', ins_tmp)

    #zero-shot
    if(len(fewpromp)==0):
      messages = [
        {"role": "system", "content": intro},
        {"role": "user", "content": ins_tmp},
      ]
    else:
      messages = [
          {"role": "system", "content": intro},
          {"role": "assistant", "content": fewpromp},
          {"role": "user", "content": ins_tmp},
      ]
    #prompt_str=intro+fewpromp+ins_tmp
    completion=completion_with_backoff(model=args.llm, messages=messages,temperature=0)
    with open(args.save_dir_filename, 'a') as file:
      file.write(str(completion.choices[0].message.content) + "\n")


