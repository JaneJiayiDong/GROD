from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import os
import anndata
import scanpy as sc
from src.utils.misc import get_dataset_information
import sklearn.metrics as metrics
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_samples
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import silhouette_score
from itertools import permutations

from src.utils.misc import create_adata

def eval_EP_info(ref_path, pred_path):
    output = pd.read_csv(pred_path, sep=',', header=0, index_col=0)
    
    output['EdgeWeight'] = abs(output['EdgeWeight'])

    output['Gene1'] = output['Gene1'].astype(str)
    output['Gene2'] = output['Gene2'].astype(str)
    output = output.sort_values('EdgeWeight',ascending=False)
    label = pd.read_csv(ref_path, sep=',', index_col=None, header=0)
    label['Gene1'] = label['Gene1'].astype(str)
    label['Gene2'] = label['Gene2'].astype(str)
    TFs = set(label['Gene1']) 
    Genes = set(label['Gene1']) | set(label['Gene2'])
    output = output[output['Gene1'].apply(lambda x: x in TFs)]
    output = output[output['Gene2'].apply(lambda x: x in Genes)]
    label_set = set(label['Gene1']+'|'+label['Gene2'])
    output= output.iloc[:len(label_set)]  # top K predicted edges
    print("k", len(label_set))
    print("possible edge", len(TFs)*len(Genes)-len(TFs))
    EPratio= len(set(output['Gene1']+ '|' +output['Gene2']) & label_set) / (len(label_set)**2/(len(TFs)*len(Genes)-len(TFs)))
    EP = len(set(output['Gene1']+ '|' +output['Gene2']) & label_set) / len(label_set) 
    return EPratio, EP

def eval_AUPR_info(ref_path, pred_path):
    output = pd.read_csv(pred_path, sep=',', header=0, index_col=0)
    output['EdgeWeight'] = abs(output['EdgeWeight'])

    output['Gene1'] = output['Gene1'].astype(str)
    output['Gene2'] = output['Gene2'].astype(str)
    output = output.sort_values('EdgeWeight',ascending=False)
    label = pd.read_csv(ref_path, sep=',', index_col=None, header=0)
    label['Gene1'] = label['Gene1'].astype(str)
    label['Gene2'] = label['Gene2'].astype(str)

    TFs = set(label['Gene1'])
    Genes = set(label['Gene1'])| set(label['Gene2']) - set("AVG")
    # print(len(Genes))
    output = output[output['Gene1'].apply(lambda x: x in TFs)]
    output = output[output['Gene2'].apply(lambda x: x in Genes)]
    label_set = set(label['Gene1']+label['Gene2'])
    # print(len(label_set))

    preds, labels, randoms = [], [], []
    res_d = {}
    l = []
    p= []
    for item in (output.to_dict('records')):
            res_d[item['Gene1']+item['Gene2']] = item['EdgeWeight']
    for item in (set(label['Gene1'])):
            for item2 in  set(label['Gene1'])| set(label['Gene2']):
                if item == item2: # TF * TG - TF
                    continue
                if item+item2 in label_set:
                    l.append(1) 
                else:
                    l.append(0)  
                if item+ item2 in res_d:
                    p.append(res_d[item+item2])
                else:
                    p.append(-1)
    print("possible edge", len(p))
    AUPRratio = average_precision_score(l,p)/np.mean(l) # 等效于网络密度
    AUPR = average_precision_score(l,p,pos_label=1)
    roc = roc_auc_score(l,p)
    return AUPRratio, AUPR, roc



def adjust_range(y):
    """Assures that the range of indices if from 0 to n-1."""
    y = np.array(y, dtype=np.int64)

    val_set = set(y)
    mapping = {val: i for i, val in enumerate(val_set)}
    y = np.array([mapping[val] for val in y], dtype=np.int64)
    return y

def hungarian_match(y_true, y_pred):
    """Matches predicted labels to original using hungarian algorithm."""

    y_true = adjust_range(y_true)
    y_pred = adjust_range(y_pred)

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    # Confusion matrix.
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(-w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    d = {i: j for i, j in ind}
    y_pred = np.array([d[v] for v in y_pred])

    return y_true, y_pred

def set_scores(scores, y_true, y_pred, scoring):
    labels=list(set(y_true))
    
    for metric in scoring:
        if metric=='accuracy':
            scores[metric] = metrics.accuracy_score(y_true, y_pred)
        # elif metric=='precision':
        #     scores[metric] = metrics.precision_score(y_true, y_pred, labels, average='macro')
        # elif metric=='recall':
        #     scores[metric] = metrics.recall_score(y_true, y_pred, labels, average='macro')
        elif metric=='f1_score':
            scores[metric] = metrics.f1_score(y_true, y_pred, average='macro')
        elif metric=='nmi':
            scores[metric] = metrics.normalized_mutual_info_score(y_true, y_pred)
        elif metric=='adj_mi':
            scores[metric] = metrics.adjusted_mutual_info_score(y_true, y_pred)
        elif metric=='adj_rand':
            scores[metric] = metrics.adjusted_rand_score(y_true, y_pred)


def compute_scores_for_cls(y_true, y_pred, scoring={'accuracy', 'precision', 'recall', 'nmi',
                                            'adj_rand', 'f1_score', 'adj_mi'}):

    scores = {}
    y_true, y_pred = hungarian_match(y_true, y_pred)
    set_scores(scores, y_true, y_pred, scoring)

    return scores

def euclidean_distance(x):
    """
    Compute euclidean distance of a tensor
    """
    n = x.shape[0]
    matrix = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            matrix[i][j] = np.square(x[i]-x[j]).mean() 

    return matrix

def cosine_distance(x):
    """
    Compute cosine distance of a tensor
    """
    n = x.shape[0]
    matrix = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            matrix[i][j] = x[i].dot(x[j]) / (np.linalg.norm(x[i]) * np.linalg.norm(x[j]))

    return matrix

def startwith(start: int, mgraph: list) -> list: # 最短路径
    # 存储已知最小长度的节点编号 即是顺序
    passed = [start]
    nopass = [x for x in range(len(mgraph)) if x != start]

    dis = mgraph[start]

    # 创建字典 为直接与start节点相邻的节点初始化路径
    dict_ = {}
    for i in range(len(dis)):
        if dis[i] != np.inf:
            dict_[str(i)] = [start]
    while len(nopass):
        idx = nopass[0]
        for i in nopass:
            if dis[i] < dis[idx]: 
                idx = i

        nopass.remove(idx)
        passed.append(idx)

        for i in nopass:
            if dis[idx] + mgraph[idx][i] < dis[i]:
                dis[i] = dis[idx] + mgraph[idx][i]
                dict_[str(i)] = dict_[str(idx)] + [idx]

    return dis,dict_

def starttoend(start: int, end: int, mgraph: list) -> list: # 经过所有点的最短路径
    # 存储已知最小长度的节点编号 即是顺序
    
    nopass = [x for x in range(len(mgraph)) if x != start and x!=end]
    dis = mgraph[start]
    
    data = list(permutations([i for i in range(start+1, end)]))
    
    res_list = []

    for d in data:
        res = 0
        for j in range(len(d) + 1):
            if j == 0:
                res += mgraph[start][d[j]]
            elif j == len(d):
                res += mgraph[d[j-1]][end]
            else:
                res += mgraph[d[j-1]][d[j]]
        res_list.append(res)
    return data, res_list


def infer_eval(network, data_dir):
    network.to_csv(data_dir + "/rankedEdge_MLP40.csv")
    for net in ['/Spec-network.csv', '/Non-Spec-network.csv', 
                '/Spec-network.csv', '/STR-network.csv', 
                '/lofgof-network.csv', '/network.csv']:
        ref_path = data_dir + net
        if not os.path.exists(ref_path):
            continue
        pred_path = data_dir +'/rankedEdge_MLP40.csv'
        EPratio, EP = eval_EP_info(ref_path, pred_path)
        AUPRratio, AUPR, roc = eval_AUPR_info(ref_path, pred_path)
        print(np.round(EPratio,4))
        print(np.round(EP,4))
        print(np.round(AUPRratio,4))
        print(np.round(AUPR,4))
        print(np.round(roc,4))
    return EPratio, EP, AUPRratio, AUPR, roc 
        
def pred_eval(pred_data, raw_data, cell_name_list, data_name, seed):
    adata = create_adata(pred_data, raw_data)
    
    adata.obs_names = cell_name_list
    split_str, point, replace_dic, cls_num = get_dataset_information(data_name)
    t_list = adata.obs_names.to_series().apply(lambda x:x.split(split_str)[point])
    
    adata.obs['time'] = t_list
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.umap(adata)
    sc.pl.umap(adata,
    color=['time'],
    wspace=0.4,
    legend_fontsize=14,
    save= data_name + 'pred_data_'+ str(seed) + '.png',
    frameon=False)

    adata.obs['time'] = adata.obs['time'].replace(replace_dic)
    y_pred = KMeans(n_clusters=cls_num).fit_predict(adata.X)
    scores_kmeans = compute_scores_for_cls(list(adata.obs['time']), y_pred)
    
    print("KMEANS:", scores_kmeans['accuracy'], scores_kmeans['f1_score'], scores_kmeans['adj_rand'], scores_kmeans['nmi'])
        
    # new matrics
    s = silhouette_score(adata.X, adata.obs['time'])
    print("silhouette_score: ", s)
    # new matrics
    l = list(set(adata.obs['time']))
    l.sort()
    centers = [adata.X[adata.obs['time'] == i].mean(0) for i in l]

    dist = euclidean_distance(np.array(centers))
    print(dist)
    dis, dict_ = startwith(0, dist)
    print(dis)
    print(dict_)

    data, res = starttoend(0, cls_num-1, dist)
    print(data)
    print(res)

    dist = cosine_distance(np.array(centers))
    print(dist)
    dis, dict_ = startwith(0, dist)
    print(dis)
    print(dict_)

    data, res = starttoend(0, cls_num-1, dist)
    print(data)
    print(res)
    print("p:", np.exp(res) / np.sum(np.exp(res)) * len(res))

    return scores_kmeans