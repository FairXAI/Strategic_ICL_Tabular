

from class_icl import *
from fcg import *
import pandas as pd
import pickle
from ..pipeLLM import askGPT,generate_kshot_template,clean_eval_labels


# in args, define the save path and dataset type
def each_group_select_k(args, client, groups, gth, n_samples, find_times, baseline_ma, xtst):
    cur_group = groups[gth]
    for i in range(find_times):
        save_dir_filename = f'res/res_k{n_samples}_round{i}_group{gth}.txt'
        cur_sample_index_set = cur_group.roulette_wheel_selection(n_samples)

        pd.DataFrame(cur_sample_index_set).to_csv(
            f'res/selectHistoryIndex/index_set_round{i}_group{gth}_cluster{cluster_num}_neigbors{m}.csv')

        merged_trs = pd.concat([X_trn, y_trn], axis=1)
        selected_dfx = get_sub_df_by_index_set(merged_trs, cur_sample_index_set)
        intro, few_prompt = generate_kshot_template(selected_dfx)

        askGPT(args, client, intro, few_prompt, xtst)
        ypred = pd.read_csv(save_dir_filename, header=None)

        bin_ypre, bin_ydev, sens_dev = clean_eval_labels(ypred, ydev, dfdev)

        new_score = cal_node_score(bin_ypre, bin_ydev, sens_dev, 0.5, baseline_ma, "F1", "EOR")
        # print(new_score)
        cur_group = update_node_scores(cur_group, init_nodes, cur_sample_index_set, new_score)
        #Pending. save the updated results [cur_group], define the save path and filenames.
    return cur_group


def create_nodes_from_index_set(each_index_set, datax, datay, zname, score):
    nodes = []
    cnt = 0
    for each_index in each_index_set:
        cnt = cnt + 1
        curx = datax.loc[each_index]
        cury = datay.loc[each_index]
        node = Node(curx, cury, curx[zname], each_index, score)
        nodes.append(node)
    return nodes


if __name__ == '__main__':
    ############ Data loading############
    #The data reading part needs to be updated
    #The process is the same as the previous demo. Eg. can be implemented by loading the data using loadAdult()/loadCredit()/ etc., in dataprocess.py
    #Note that dev is used for eval in FCG stages. So split from train data -> dftr=[dev,tr]
    dfdev=''#dataframe dev df, read from dataset used in FCG eval
    xdev=''#dataframe dev df.x, read from dataset used in FCG eval
    ydev=''#dataframe dev df.y, read from dataset used in FCG eval
    X_trn=''#dataframe train df.x, read from dataset
    y_trn=''#dataframe train df.y, read from dataset
    zname=''#dataframe sensitive feature
    ###################################Loading finished################################################

    nodes=[]
    for i in range(len(X_trn)):
        curx=X_trn.iloc[i]
        cury=y_trn.iloc[i]
        node = Node(curx, cury, curx[zname],curx.name, 0)
        nodes.append(node)

    ############Step1. Clustering############
    cluster_num=8
    m=5
    #use klearn for clustering, refer to 'run_kmeans','cluster_centers_indices' in fcg.py
    #save clustered indices to all_indices after kmeans clustering
    all_indices=[]

    ############Step2. genetic selection############
    init_score=0.05
    init_nodes=create_nodes_from_index_set(all_indices,X_trn,y_trn,zname,init_score)
    group_FH = NodeGroup(y=">50K", z="Female",nodelist=init_nodes)
    group_FL = NodeGroup(y="<=50K", z="Female",nodelist=init_nodes)
    group_MH = NodeGroup(y=">50K", z="Male",nodelist=init_nodes)
    group_ML = NodeGroup(y="<=50K", z="Male",nodelist=init_nodes)
    groups=[group_FH,group_FL,group_MH,group_ML]


    k_samples=-1#set parameters
    find_times=-1#set parameters
    epoch=-1 #set parameters
    #define the baseline for acc,f1,dpr,eor \in [0,1]. Eg. acc/f1:0.5~0.8, dpr/eor:0.2~0.6
    baseline_ma=NodeMetrics(acc=-1,f1=-1,dpr=-1,eor=-1)


    for i in range(1,epoch):
      each_group_select_k(groups,i,k_samples,find_times,baseline_ma)
