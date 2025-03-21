
from functools import partial
from scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.stats import rv_discrete, multivariate_normal, multivariate_t, wishart, uniform
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from numba.experimental import jitclass
from numba import int32, float32
from numba.typed import List

class CoBaFi:
    def __init__(self, scores, d, nu, gamma, delta):
        self.d = d
        self.nu_ini = nu
        self.gamma = gamma
        self.delta = delta

        self.user_num = scores[:, 0].max() + 1
        self.item_num = scores[:, 1].max() + 1
        self.R = scores[:, 2].max()
        self.phi = np.array([[r - self.R / 2.0, (r - self.R / 2.0) ** 2] for r in range(1, self.R + 1)])

        self.scores = scores
        self.user_scores = [[] for _ in range(self.user_num)]
        self.item_scores = [[] for _ in range(self.item_num)]
        for (user_idx, item_idx, score) in scores:
            self.user_scores[user_idx].append((item_idx, score))
            self.item_scores[item_idx].append((user_idx, score))

        # user clusters
        self.assigned_user_nums = np.zeros(0, dtype=np.int32)
        self.assigned_item_nums = np.zeros(0, dtype=np.int32)
        self.user_cluster_means = np.zeros(0, dtype=np.float32)
        self.item_cluster_means = np.zeros(0, dtype=np.float32)
        self.user_cluster_sigmas = np.zeros(0, dtype=np.float32)
        self.item_cluster_sigmas = np.zeros(0, dtype=np.float32)
        #self.user_clusters = []  # [[assigned_user_num, mean, covariance]]
        #self.item_clusters = []  # [[assigned_item_num, mean, covariance]]
        self.user_assignment = np.full(self.user_num, -1)
        self.item_assignment = np.full(self.item_num, -1)
        self.user_params = np.zeros((self.user_num, d))
        self.item_params = np.zeros((self.item_num, d))


    def negative_log_likelihood(self):
        llh = 0.0
        for (user_idx, item_idx, score) in self.scores:
            user_param = self.user_params[user_idx]
            item_param = self.item_params[item_idx]
            theta = self.compute_theta(user_param, item_param)
            llh -= (self.phi[score - 1] * theta).sum() - self.g_func(theta)
        return llh

    def init_infer(self):

        def init_sampling_cluster(all_num, clusters, gamma):
            #tables = torch.tensor([cluster[0] / (all_num + gamma) for cluster in clusters] + [gamma / (all_num + gamma)])
            #cluster_idx = pyro.sample("cluster", dist.Categorical(tables)
            tables = np.array([cluster[0] / (all_num + gamma) for cluster in clusters] + [gamma / (all_num + gamma)])
            cluster_idx = rv_discrete(values=(np.arange(len(clusters) + 1), tables)).rvs()
            return cluster_idx
        
        ### hyperparameters ###
        mu_alpha = np.zeros(self.d)
        W_alpha = np.eye(self.d)
        lambda_alpha = 1.

        mu_beta = deepcopy(mu_alpha)
        W_beta = deepcopy(W_alpha)
        lambda_beta = 1.

        ### generate first cluster ###
        user_cluster_mu, user_cluster_sigma = self.sampling_cluster_parameters_for_new_cluster(mu_alpha, W_alpha, lambda_alpha)
        item_cluster_mu, item_cluster_sigma = self.sampling_cluster_parameters_for_new_cluster(mu_beta, W_beta, lambda_beta)
        #user_cluster_sigma = pyro.sample('user_cluster_sigma', dist.Wishart(self.nu_ini, W_alpha))
        #user_cluster_mu = pyro.sample('user_cluster_mu', dist.MultivariateNormal(mu_alpha, user_cluster_sigma))
        #item_cluster_sigma = pyro.sample('item_cluster_sigma', dist.Wishart(self.nu_ini, W_beta))
        #item_cluster_mu = pyro.sample('item_cluster_mu', dist.MultivariateNormal(mu_beta, item_cluster_sigma))

        self.user_clusters.append([1, user_cluster_mu, user_cluster_sigma])
        self.item_clusters.append([1, item_cluster_mu, item_cluster_sigma])
        self.user_assignment[0] = 0
        self.item_assignment[0] = 0
        self.user_params[0] = self.sampling_parameters(user_cluster_mu, user_cluster_sigma)
        self.item_params[0] = self.sampling_parameters(item_cluster_mu, item_cluster_sigma)

        ### initial clustering users and items (Chinese Restaurant Process) and initialize user and item parameters ###
        for user_idx in range(1, self.user_num):
            #user_cluster_idx = init_sampling_cluster(user_idx, self.user_clusters, self.gamma)
            user_cluster_idx = user_idx
            self.user_assignment[user_idx] = user_cluster_idx
            if user_cluster_idx == len(self.user_clusters):
                # for new user cluster
                user_cluster_mu, user_cluster_sigma = self.sampling_cluster_parameters_for_new_cluster(mu_alpha, W_alpha, lambda_alpha)
                self.user_clusters.append([1, user_cluster_mu, user_cluster_sigma])
            else:
                # for existing user cluster
                _, user_cluster_mu, user_cluster_sigma = self.user_clusters[user_cluster_idx]
                self.user_clusters[user_cluster_idx][0] += 1
            self.user_params[user_idx] = self.sampling_parameters(user_cluster_mu, user_cluster_sigma)

        for item_idx in range(1, self.item_num):
            #item_cluster_idx = init_sampling_cluster(item_idx, self.item_clusters, self.delta)
            item_cluster_idx = item_idx
            self.item_assignment[item_idx] = item_cluster_idx
            if item_cluster_idx == len(self.item_clusters):
                # for new item cluster
                item_cluster_mu, item_cluster_sigma = self.sampling_cluster_parameters_for_new_cluster(mu_beta, W_beta, lambda_beta)
                self.item_clusters.append([1, item_cluster_mu, item_cluster_sigma])
            else:
                # for existing item cluster
                _, item_cluster_mu, item_cluster_sigma = self.item_clusters[item_cluster_idx]
                self.item_clusters[item_cluster_idx][0] += 1
            self.item_params[item_idx] = self.sampling_parameters(item_cluster_mu, item_cluster_sigma)

        ### sampling user and item parameters ###
        self.sampling_user_parameters()
        self.sampling_item_parameters()

        ### clustering (collapsed Gibbs sampling) ###
        self.user_clusters = []
        self.item_clusters = []
        self.user_assignment = np.full(self.user_num, -1)
        self.item_assignment = np.full(self.item_num, -1)
        self.clustering_users(mu_alpha, W_alpha, lambda_alpha)
        self.clustering_items(mu_beta, W_beta, lambda_beta)

        ### sampling user and item cluster parameters ###
        '''for user_cluster_idx, (assigned_user_num, _, _) in enumerate(self.user_clusters):
            user_params_in_cluster = self.user_params[self.user_assignment == user_cluster_idx]
            self.user_clusters[user_cluster_idx][1:] = self.sampling_cluster_parameters_for_existing_cluster(mu_alpha, W_alpha, lambda_alpha, assigned_user_num, user_params_in_cluster)

        for item_cluster_idx, (assigned_item_num, _, _) in enumerate(self.item_clusters):
            item_params_in_cluster = self.item_params[self.item_assignment == item_cluster_idx]
            self.item_clusters[item_cluster_idx][1:] = self.sampling_cluster_parameters_for_existing_cluster(mu_beta, W_beta, lambda_beta, assigned_item_num, item_params_in_cluster)
        
        self.sampling_user_parameters()
        self.sampling_item_parameters()'''

    def update_infer(self, h=1e-3):

        ### hyperparameters ###
        user_cluster_inv_sigmas = [np.linalg.inv(user_cluster_sigma) for _, _, user_cluster_sigma in self.user_clusters]
        item_cluster_inv_sigmas = [np.linalg.inv(item_cluster_sigma) for _, _, item_cluster_sigma in self.item_clusters]

        mu_alpha = np.linalg.inv(sum(user_cluster_inv_sigmas)) @ sum(user_cluster_inv_sigmas[user_cluster_idx] @ user_cluster_mu for user_cluster_idx, (_, user_cluster_mu, _) in enumerate(self.user_clusters))
        W_alpha = len(self.user_clusters) * (self.nu_ini - self.d - 1) * np.linalg.inv(sum(user_cluster_sigma for _, _, user_cluster_sigma in self.user_clusters))
        lambda_alpha = sum((user_cluster_mu - mu_alpha) @ user_cluster_inv_sigmas[user_cluster_idx] @ (user_cluster_mu - mu_alpha).T for user_cluster_idx, (_, user_cluster_mu, _) in enumerate(self.user_clusters)) / (len(self.user_clusters) * self.d)

        mu_beta = np.linalg.inv(sum(item_cluster_inv_sigmas)) @ sum(item_cluster_inv_sigmas[item_cluster_idx] @ item_cluster_mu for item_cluster_idx, (_, item_cluster_mu, _) in enumerate(self.item_clusters))
        W_beta = len(self.item_clusters) * (self.nu_ini - self.d - 1) * np.linalg.inv(sum(item_cluster_sigma for _, _, item_cluster_sigma in self.item_clusters))
        lambda_beta = sum((item_cluster_mu - mu_beta) @ item_cluster_inv_sigmas[item_cluster_idx] @ (item_cluster_mu - mu_beta).T for item_cluster_idx, (_, item_cluster_mu, _) in enumerate(self.item_clusters)) / (len(self.item_clusters) * self.d)

        ### sampling user and item cluster parameters ###
        for user_cluster_idx, (assigned_user_num, _, _) in enumerate(self.user_clusters):
            user_params_in_cluster = self.user_params[self.user_assignment == user_cluster_idx]
            self.user_clusters[user_cluster_idx][1:] = self.sampling_cluster_parameters_for_existing_cluster(mu_alpha, W_alpha, lambda_alpha, assigned_user_num, user_params_in_cluster)

        ### clustering (collapsed Gibbs sampling) ###
        self.clustering_users(mu_alpha, W_alpha, lambda_alpha)

        ### sampling user and item parameters ###
        self.sampling_user_parameters()

        for item_cluster_idx, (assigned_item_num, _, _) in enumerate(self.item_clusters):
            item_params_in_cluster = self.item_params[self.item_assignment == item_cluster_idx]
            self.item_clusters[item_cluster_idx][1:] = self.sampling_cluster_parameters_for_existing_cluster(mu_beta, W_beta, lambda_beta, assigned_item_num, item_params_in_cluster)

        self.clustering_items(mu_beta, W_beta, lambda_beta)
        self.sampling_item_parameters()


    @staticmethod
    def compute_theta(user_param, item_param):
        return np.array([user_param[0] + item_param[0] + (user_param[2:] * item_param[2:]).sum(), user_param[1] + item_param[1]])


    def g_func(self, theta):
        return logsumexp((self.phi * theta).sum(axis=1))
        #return np.log(np.exp((self.phi * theta).sum(axis=1)).sum())


    @staticmethod
    def sampling_parameters(mu, sigma):
        a = sigma.diagonal()
        is_symmetric = np.allclose(sigma, sigma.T)
        #return pyro.sample('params', dist.MultivariateNormal(mu, sigma))
        return multivariate_normal(mu, sigma).rvs()
    

    def sampling_user_parameters(self, h=1e-3):
        user_cluster_inv_sigmas = [np.linalg.inv(user_cluster_sigma) for _, _, user_cluster_sigma in self.user_clusters]

        def negative_log_user_param_posterior(user_param, user_idx):
            user_cluster_idx = self.user_assignment[user_idx]
            _, user_cluster_mu, _ = self.user_clusters[user_cluster_idx]
            user_cluster_inv_sigma = user_cluster_inv_sigmas[user_cluster_idx]

            lupp = 0.0
            for item_idx, score in self.user_scores[user_idx]:
                item_param = self.item_params[item_idx]
                theta = self.compute_theta(user_param, item_param)
                lupp -= (self.phi[score - 1] * theta).sum() - self.g_func(theta)

            return lupp + (user_param - user_cluster_mu) @ user_cluster_inv_sigma @ (user_param - user_cluster_mu).T / 2.0

        for user_idx in tqdm(range(self.user_num), desc="sampling user parameters", postfix='range'):
            user_param = self.user_params[user_idx]

            # sup: sampling user parameters
            #sup_mu = minimize(partial(negative_log_user_param_posterior, user_idx=user_idx), self.user_params[user_idx], options={'maxiter': self.d * 2}).x
            sup_mu = minimize(partial(negative_log_user_param_posterior, user_idx=user_idx), self.user_params[user_idx], options={'maxiter': self.d * 2}).x
            inv_sup_sigma = np.linalg.inv(self.user_clusters[self.user_assignment[user_idx]][2])

            for item_idx, _ in self.user_scores[user_idx]:
                item_param = self.item_params[item_idx]

                cur_g_value = self.g_func(self.compute_theta(user_param, item_param))

                for i in range(self.d):
                    ph_user_param = deepcopy(user_param)
                    mh_user_param = deepcopy(user_param)
                    ph_user_param[i] += h
                    mh_user_param[i] -= h
                    ph_g_value = self.g_func(self.compute_theta(ph_user_param, item_param))
                    mh_g_value = self.g_func(self.compute_theta(mh_user_param, item_param))

                    inv_sup_sigma[i][i] += (ph_g_value - 2 * cur_g_value + mh_g_value) / (h ** 2.0)

                for i in range(self.d):
                    for j in range(i + 1, self.d):
                        pph_user_param = deepcopy(user_param)
                        pmh_user_param = deepcopy(user_param)
                        mph_user_param = deepcopy(user_param)
                        mmh_user_param = deepcopy(user_param)
                        pph_user_param[i] += h; pph_user_param[j] += h
                        pmh_user_param[i] += h; pmh_user_param[j] -= h
                        mph_user_param[i] -= h; mph_user_param[j] += h
                        mmh_user_param[i] -= h; mmh_user_param[j] -= h
                        pph_g_value = self.g_func(self.compute_theta(pph_user_param, item_param))
                        pmh_g_value = self.g_func(self.compute_theta(pmh_user_param, item_param))
                        mph_g_value = self.g_func(self.compute_theta(mph_user_param, item_param))
                        mmh_g_value = self.g_func(self.compute_theta(mmh_user_param, item_param))

                        d_value = (pph_g_value - pmh_g_value - mph_g_value + mmh_g_value) / (4 * h ** 2.0)
                        inv_sup_sigma[i][j] += d_value
                        inv_sup_sigma[j][i] += d_value

            try:
                sup_sigma = np.linalg.inv(inv_sup_sigma)
                new_user_param = self.sampling_parameters(sup_mu, sup_sigma)
            except:
                sup_sigma = self.user_clusters[self.user_assignment[user_idx]][2]
                new_user_param = self.sampling_parameters(sup_mu, sup_sigma)

            log_mh_accept_prob = negative_log_user_param_posterior(user_param, user_idx) - multivariate_normal(sup_mu, sup_sigma).logpdf(new_user_param) \
                               - negative_log_user_param_posterior(new_user_param, user_idx) + multivariate_normal(sup_mu, sup_sigma).logpdf(user_param)
            
            if log_mh_accept_prob >= 0.0 or np.exp(log_mh_accept_prob) > uniform.rvs():
                self.user_params[user_idx] = new_user_param


    def sampling_item_parameters(self, h=1e-3):
        item_cluster_inv_sigmas = [np.linalg.inv(item_cluster_sigma) for _, _, item_cluster_sigma in self.item_clusters]

        def negative_log_item_param_posterior(item_param, item_idx):
            item_cluster_idx = self.item_assignment[item_idx]
            _, item_cluster_mu, _ = self.item_clusters[item_cluster_idx]
            item_cluster_inv_sigma = item_cluster_inv_sigmas[item_cluster_idx]

            lupp = 0.0
            for user_idx, score in self.item_scores[item_idx]:
                user_param = self.user_params[user_idx]
                theta = self.compute_theta(user_param, item_param)
                lupp -= (self.phi[score - 1] * theta).sum() - self.g_func(theta)

            return lupp + (item_param - item_cluster_mu) @ item_cluster_inv_sigma @ (item_param - item_cluster_mu).T / 2.0

        
        for item_idx in tqdm(range(self.item_num), desc="sampling item parameters", postfix='range'):
            item_param = self.item_params[item_idx]

            # sip: sampling item parameters
            sip_mu = minimize(partial(negative_log_item_param_posterior, item_idx=item_idx), self.item_params[item_idx], options={'maxiter': self.d * 2}).x
            inv_sip_sigma = np.linalg.inv(self.item_clusters[self.item_assignment[item_idx]][2])

            for user_idx, _ in self.item_scores[item_idx]:
                user_param = self.user_params[user_idx]

                cur_g_value = self.g_func(self.compute_theta(user_param, item_param))

                for i in range(self.d):
                    ph_item_param = deepcopy(item_param)
                    mh_item_param = deepcopy(item_param)
                    ph_item_param[i] += h
                    mh_item_param[i] -= h
                    ph_g_value = self.g_func(self.compute_theta(user_param, ph_item_param))
                    mh_g_value = self.g_func(self.compute_theta(user_param, mh_item_param))

                    inv_sip_sigma[i][i] += (ph_g_value - 2 * cur_g_value + mh_g_value) / (h ** 2.0)

                for i in range(self.d):
                    for j in range(i + 1, self.d):
                        pph_item_param = deepcopy(item_param)
                        pmh_item_param = deepcopy(item_param)
                        mph_item_param = deepcopy(item_param)
                        mmh_item_param = deepcopy(item_param)
                        pph_item_param[i] += h; pph_item_param[j] += h
                        pmh_item_param[i] += h; pmh_item_param[j] -= h
                        mph_item_param[i] -= h; mph_item_param[j] += h
                        mmh_item_param[i] -= h; mmh_item_param[j] -= h
                        pph_g_value = self.g_func(self.compute_theta(user_param, pph_item_param))
                        pmh_g_value = self.g_func(self.compute_theta(user_param, pmh_item_param))
                        mph_g_value = self.g_func(self.compute_theta(user_param, mph_item_param))
                        mmh_g_value = self.g_func(self.compute_theta(user_param, mmh_item_param))

                        d_value = (pph_g_value - pmh_g_value - mph_g_value + mmh_g_value) / (4 * h ** 2.0)
                        inv_sip_sigma[i][j] += d_value
                        inv_sip_sigma[j][i] += d_value

            try:
                sip_sigma = np.linalg.inv(inv_sip_sigma)
                new_item_param = self.sampling_parameters(sip_mu, sip_sigma)
            except:
                sip_sigma = self.item_clusters[self.item_assignment[item_idx]][2]
                new_item_param = self.sampling_parameters(sip_mu, sip_sigma)

            log_mh_accept_prob = negative_log_item_param_posterior(item_param, item_idx) - multivariate_normal(sip_mu, sip_sigma).logpdf(new_item_param) \
                               - negative_log_item_param_posterior(new_item_param, item_idx) + multivariate_normal(sip_mu, sip_sigma).logpdf(item_param)
            
            if log_mh_accept_prob >= 0.0 or np.exp(log_mh_accept_prob) > uniform.rvs():
                self.item_params[item_idx] = new_item_param


    def clustering_users(self, mu_alpha, W_alpha, lambda_alpha):
        for user_idx in tqdm(range(self.user_num), desc="clustering users", postfix='range'):
            user_param = self.user_params[user_idx]

            prev_user_cluster_idx = self.user_assignment[user_idx]
            if prev_user_cluster_idx != -1:
                self.user_clusters[prev_user_cluster_idx][0] -= 1
                self.user_assignment[user_idx] = -1

                # if previous cluster is empty, then remove it
                if self.user_clusters[prev_user_cluster_idx][0] == 0:
                    self.user_clusters.pop(prev_user_cluster_idx)
                    self.user_assignment = np.where(self.user_assignment > prev_user_cluster_idx, self.user_assignment - 1, self.user_assignment)

            # snc: sampling new user cluster
            snuc_nu = self.nu_ini + self.d - 2
            snuc_mu = (lambda_alpha * mu_alpha + user_param) / (lambda_alpha + 1)
            snuc_sigma = (1/(lambda_alpha + 2) * ((lambda_alpha + 1) * snuc_nu) * (np.linalg.inv(W_alpha) + (lambda_alpha / (lambda_alpha + 1)) * np.outer(user_param - mu_alpha, user_param - mu_alpha)))

            tables = np.array([np.log(assigned_num) + multivariate_normal(cluster_mu, cluster_sigma, allow_singular=True).logpdf(user_param) for assigned_num, cluster_mu, cluster_sigma in self.user_clusters] + [np.log(self.gamma) + multivariate_t(snuc_mu, snuc_sigma, snuc_nu, allow_singular=True).logpdf(user_param)])
            tables -= tables.max()
            tables = np.exp(tables)
            tables /= tables.sum()
            user_cluster_idx = rv_discrete(values=(np.arange(len(self.user_clusters) + 1), tables)).rvs()
            self.user_assignment[user_idx] = user_cluster_idx
            if user_cluster_idx == len(self.user_clusters):
                # for new user cluster
                user_cluster_mu, user_cluster_sigma = self.sampling_cluster_parameters_for_new_cluster(mu_alpha, W_alpha, lambda_alpha)
                self.user_clusters.append([1, user_cluster_mu, user_cluster_sigma])
            else:
                # for existing user cluster
                _, user_cluster_mu, user_cluster_sigma = self.user_clusters[user_cluster_idx]
                self.user_clusters[user_cluster_idx][0] += 1


    def clustering_items(self, mu_beta, W_beta, lambda_beta):
        for item_idx in tqdm(range(self.item_num), desc="clustering items", postfix='range'):
            item_param = self.item_params[item_idx]

            prev_item_cluster_idx = self.item_assignment[item_idx]
            if prev_item_cluster_idx != -1:
                self.item_clusters[prev_item_cluster_idx][0] -= 1
                self.item_assignment[item_idx] = -1

                # if previous cluster is empty, then remove it
                if self.item_clusters[prev_item_cluster_idx][0] == 0:
                    self.item_clusters.pop(prev_item_cluster_idx)
                    self.item_assignment = np.where(self.item_assignment > prev_item_cluster_idx, self.item_assignment - 1, self.item_assignment)

            # snic: sampling new item cluster
            snic_nu = self.nu_ini + self.d - 2
            snic_mu = (lambda_beta * mu_beta + item_param) / (lambda_beta + 1)
            snic_sigma = (1/(lambda_beta + 2) * ((lambda_beta + 1) * snic_nu) * (np.linalg.inv(W_beta) + (lambda_beta / (lambda_beta + 1)) * np.outer(item_param - mu_beta, item_param - mu_beta)))

            tables = np.array([np.log(assigned_num) + multivariate_normal(cluster_mu, cluster_sigma, allow_singular=True).logpdf(item_param) for assigned_num, cluster_mu, cluster_sigma in self.item_clusters] + [np.log(self.delta) + multivariate_t(snic_mu, snic_sigma, snic_nu, allow_singular=True).logpdf(item_param)])
            tables -= tables.max()
            tables = np.exp(tables)
            tables /= tables.sum()
            item_cluster_idx = rv_discrete(values=(np.arange(len(self.item_clusters) + 1), tables)).rvs()
            self.item_assignment[item_idx] = item_cluster_idx
            if item_cluster_idx == len(self.item_clusters):
                # for new item cluster
                item_cluster_mu, item_cluster_sigma = self.sampling_cluster_parameters_for_new_cluster(mu_beta, W_beta, lambda_beta)
                self.item_clusters.append([1, item_cluster_mu, item_cluster_sigma])
            else:
                # for existing item cluster
                _, item_cluster_mu, item_cluster_sigma = self.item_clusters[item_cluster_idx]
                self.item_clusters[item_cluster_idx][0] += 1


    def sampling_cluster_parameters_for_new_cluster(self, mu, W, _lambda):
        #cluster_sigma = pyro.sample('cluster sigma', dist.Wishart(self.nu_ini, W))
        #cluster_mu = pyro.sample('cluster mu', dist.MultivariateNormal(mu, _lambda * cluster_sigma))
        cluster_sigma = wishart(self.nu_ini, W).rvs()
        cluster_mu = multivariate_normal(mu, np.linalg.inv(_lambda * cluster_sigma)).rvs()
        return cluster_mu, cluster_sigma


    def sampling_cluster_parameters_for_existing_cluster(self, mu, W, _lambda, assigned_num, params_in_cluster):
        # scp: sampling cluster parameters
        scp_lambda = _lambda + assigned_num
        scp_mu = ((_lambda * mu) + params_in_cluster.sum(axis=0)) / scp_lambda
        #scp_W = np.linalg.inv((_lambda * np.linalg.inv(W) + _lambda * np.outer(mu, mu) + sum(np.outer(param, param) for param in params_in_cluster)) / scp_lambda)
        mean_params = params_in_cluster.mean(axis=0)
        scp_W = np.linalg.inv(np.linalg.inv(W) + sum(np.outer(param - mean_params, param - mean_params) for param in params_in_cluster) + assigned_num * _lambda / scp_lambda * np.outer(mean_params - scp_mu, mean_params - scp_mu))
        scp_nu = self.nu_ini + assigned_num

        #cluster_sigma = pyro.sample('cluster sigma', dist.Wishart(scp_nu, scp_W))
        #cluster_mu = pyro.sample('cluster mu', dist.MultivariateNormal(scp_mu, scp_lambda * cluster_sigma))
        cluster_sigma = wishart(scp_nu, scp_W).rvs()
        cluster_mu = multivariate_normal(scp_mu, np.linalg.inv(scp_lambda * cluster_sigma)).rvs()
        return cluster_mu, cluster_sigma