import argparse
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import preprocessing

from cobafi import CoBaFi


def load_data(dataset_name, user_col, item_col, score_col, spam_col):
    data = pd.read_csv(f"dataset/{dataset_name}.csv.gz", nrows=100000)

    oe = preprocessing.OrdinalEncoder()
    data[[user_col, item_col]] = oe.fit_transform(data[[user_col, item_col]]).astype(int)

    user_dict = {category: idx for idx, category in enumerate(oe.categories_[0])}
    item_dict = {category: idx for idx, category in enumerate(oe.categories_[1])}

    return data[[user_col, item_col, score_col, spam_col]], user_dict, item_dict


parser = argparse.ArgumentParser()

# dataset info
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--user_col', type=str)
parser.add_argument('--item_col', type=str)
parser.add_argument('--score_col', type=str)
parser.add_argument('--spam_col', type=str)

# hyperparameters
parser.add_argument('--latent_useritem_params_num', type=int, default=20)
parser.add_argument('--new_user_cluster_param', type=float, default=0.1)
parser.add_argument('--new_item_cluster_param', type=float, default=0.1)

# inference options
parser.add_argument('--max_iter', type=int, default=100)

args = parser.parse_args()


user_col = args.user_col
item_col = args.item_col
score_col = args.score_col
spam_col = args.spam_col

d = args.latent_useritem_params_num
nu = d * 1.5
gamma = args.new_user_cluster_param
delta = args.new_item_cluster_param

max_iter = args.max_iter

if d < 2:
    print('"latent_useritem_param_num must be equal or more than 2"')
    exit(0)

data, user_dict, item_dict = load_data(args.dataset_name, user_col, item_col, score_col, spam_col)
scores = np.array(data[[user_col, item_col, score_col]])

spam_users = data[data[spam_col] == 1][user_col].unique()
print(f"num of spam users: {len(spam_users)}")

model = CoBaFi(scores, d=d, nu=nu, gamma=gamma, delta=delta)
model.init_infer()
print(f"initial negative log likelihood: {model.negative_log_likelihood()}")

for iter in range(1, max_iter + 1):
    model.update_infer()
    print(f"negative log likelihood after iter {iter}: {model.negative_log_likelihood()}")
    print(len(model.user_clusters))
    print(len(model.item_clusters))

    cluster_user_nums = np.zeros(len(model.user_clusters))
    cluster_spam_ratios = np.zeros(len(model.user_clusters))
    for spam_user in spam_users:
        cluster_spam_ratios[model.user_assignment[spam_user]] += 1

    for user_cluster_id, (assigned_user_num, _, _) in enumerate(model.user_clusters):
        cluster_user_nums[user_cluster_id] = assigned_user_num
        cluster_spam_ratios[user_cluster_id] /= assigned_user_num

    cluster_spam_ratios_df = pd.DataFrame({'cluster': list(range(1, len(model.user_clusters) + 1)), 'innocent num': cluster_user_nums * (1 - cluster_spam_ratios), 'spam num': cluster_user_nums * cluster_spam_ratios})
    fig = px.bar(cluster_spam_ratios_df, x='cluster', y=['innocent num', 'spam num'])
    fig.write_html(f"spam_ratio_after iter{iter}.html")