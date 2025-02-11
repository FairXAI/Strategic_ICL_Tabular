from sklearn.cluster import KMeans
import numpy as np
from fairlearn.metrics import equalized_odds_difference,equalized_odds_ratio,demographic_parity_ratio,demographic_parity_difference
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_kmeans(data,k,sd):
  kmeans = KMeans(n_clusters=k,random_state=sd)
  kmeans.fit(data)
  labels = kmeans.labels_
  centroids = kmeans.cluster_centers_
  return labels,centroids

#m: select m samples in each cluster
def select_kmeans_points_Index(data,centroids,m):
  cluster_centers_indices = []
  for center in centroids:
      distances = np.linalg.norm(data - center, axis=1)
      #closest_index = np.argmin(distances)[:m]#point w min distance
      closest_index = np.argsort(distances)[:m]
      cluster_centers_indices.extend(closest_index) #index
  return cluster_centers_indices


def cal_node_score(bin_ypre,bin_ytst,sens,ratio,baseline_metrics,option_pred,option_fair):
  delta_pred=0
  delta_fair=0
  if option_pred=="ACC":
    ACC=accuracy_score(bin_ytst, bin_ypre)
    delta_pred=max(ACC - baseline_metrics.acc, 0.05)
  elif option_pred=="F1":
    F1=f1_score(bin_ytst, bin_ypre, average='binary')
    #print(f"F1{F1}")
    delta_pred=max(F1 - baseline_metrics.f1, 0.05)

  if option_fair=="DPR":
    DPR=demographic_parity_ratio(bin_ytst, bin_ypre,sensitive_features=sens)
    delta_fair=max(DPR - baseline_metrics.dpr, 0.05)
  elif option_pred=="EOR":
    EOR=equalized_odds_ratio(bin_ytst, bin_ypre, sensitive_features=sens)
    delta_fair=max(EOR - baseline_metrics.eor, 0.05)

  res=2*( ratio*delta_pred + (1-ratio)*delta_fair )
  print(f"fair:{delta_fair}, delta pred:{delta_pred}, res:{res}")
  return res


def update_node_scores(node_group, node_list, indices_to_update, new_score):
    for node_index in indices_to_update:
        # Find node instance in the given list
        node = next((n for n in node_list if n.index == node_index), None)
        if node:
            node.update_rank(new_score)
            if node.label_y == node_group.label_y and node.label_z == node_group.label_z:
                idx = node_group.index_set.index(node_index)
                node_group.avg_rank_set[idx] = node.avg_rank_score
    return node_group


#convert org. index_set to the actual pos.
def convert_pos_to_global_index(dataframe, index_set):
    global_index=[]
    for pos_index in index_set:
      cur_index=dataframe.iloc[pos_index].name
      global_index.append(cur_index)
    return global_index

#return df for the given index_set
def get_sub_df_by_index_set(dataframe, index_set):
    sub_dataframe = dataframe.loc[index_set]
    return sub_dataframe


