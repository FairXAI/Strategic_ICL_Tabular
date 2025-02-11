import random
import numpy as np


class NodeMetrics:
    def __init__(self, acc, f1, dpr, eor):
        self.acc = float(acc)
        self.f1 = float(f1)
        self.dpr = float(dpr)
        self.eor = float(eor)


class Node:
    def __init__(self, x, y, z, index, raw_score):
        self.df_x = x
        self.label_y = y  # y label
        self.label_z = z  # z sensitive
        self.index = index

        self.acc_set = []  # each round
        self.fair_set = []  # each round
        self.rank_set = [raw_score]  # each round
        self.avg_rank_score = raw_score

    def update_acc_fair(self, acc_score, fair_score):
        self.acc_set.append(acc_score)
        self.fair_set.append(fair_score)

    def update_rank(self, new_score):
        self.rank_set.append(new_score)
        self.avg_rank_score = np.mean(self.rank_set)


class NodeGroup:
    def __init__(self, y, z, nodelist=None):
        self.label_y = y  # y label
        self.label_z = z  # z sensitive

        self.index_set = []
        self.avg_rank_set = []  # each nodes' avg_rank score
        if nodelist is not None:
            for node in nodelist:
                if node.label_y == self.label_y and node.label_z == self.label_z:
                    self.index_set.append(node.index)
                    self.avg_rank_set.append(node.avg_rank_score)

    def update_matching_nodes(self, node_list):
        matching_indices = []
        matching_scores = []
        for node in node_list:
            if node.label_y == self.label_y and node.label_z == self.label_z:
                matching_indices.append(node.index)
                matching_scores.append(node.avg_rank_score)
        self.index_set = matching_indices
        self.avg_rank_set = matching_scores

    def getSize(self):
        return len(self.index_set)

    def getTopK(self, k):
        score_index_pairs = list(zip(self.avg_rank_set, self.index_set))
        score_index_pairs.sort(key=lambda x: x[0], reverse=True)
        # Highest K scores' index
        top_k_indices = [index for _, index in score_index_pairs[:k]]
        return top_k_indices

    def roulette_wheel_selection(self, K):
        total_score = sum(self.avg_rank_set)
        if total_score == 0:
            return random.sample(self.index_set, K)

        # selection prob for each nodes
        selection_probabilities = [score / total_score for score in self.avg_rank_set]
        # print(selection_probabilities)

        selected_indices = np.random.choice(self.index_set, size=K, p=selection_probabilities, replace=False)
        return list(selected_indices)

