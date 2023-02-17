import numpy as np
import random
import pandas as pd

class NormalCKDEImputer:
    def __init__(self, data):
        self.epsilon = 0.00001
        self.stopping_threshold = 100
        self.data = data
        self.variance = None
        self.n_obs = self.data.shape[0]
        self.n_features = self.data.shape[1]
    
    def fit(self):
        counter = 0
        variance = 1
        variance_list =[variance]
        
        while not self._is_converged(variance_list):
            variance_old = variance_list[-1]
            variance_new = self._update_variance(variance_old)
            variance_list.append(variance_new)
            counter += 1
            if counter >= self.stopping_threshold:
                break
                
        self.variance = variance_list[-1]
        return self
        
    def _update_variance(self, variance_old):
        update_weight = 1 / (self.n_obs * self.n_features)
        observation_list = []
        
        for obs_index in range(self.n_obs):
            base_obs = self.data[obs_index, :]
            nominator = sum([np.exp(-np.linalg.norm(np.subtract(base_obs, self.data[i, :]))**2 / (2 * variance_old))
                             * np.linalg.norm(np.subtract(base_obs, self.data[i, :]))**2 
                             for i in range(self.n_obs)
                             if i != obs_index])
            denominator = sum([np.exp(-np.linalg.norm(np.subtract(base_obs, self.data[i, :]))**2 / (2 * variance_old))
                               for i in range(self.n_obs)
                               if i != obs_index])
            observation_list.append(nominator / denominator)
            
        return update_weight * sum(observation_list)
    
    def _is_converged(self, variance_list):
        variance_history = 3
        if len(variance_list) < variance_history:
            return False
        considered_elements = np.array(variance_list[-variance_history:])
        differences = np.diff(considered_elements)
        return np.sum(differences >= self.epsilon) == 0
    
    def predict(self, feature_observation, index_obs = None):
        rng = np.random.default_rng()
        # convert feature observation to numpy array
        feature_obs_arr = np.array(feature_observation, copy=True)
        # calculate weights
        # I first have to extract the indizes of given and to be imputed features
        index_given = np.where(feature_obs_arr != 0)
        index_impute = np.where(feature_obs_arr == 0) # I assume every obs to be imputed is 0
        # now calculate weights with only given indizes
        feature_obs_given = feature_obs_arr[index_given]
        nominators = [np.exp(-np.linalg.norm(feature_obs_given - observation[index_given])**2 / (2 * self.variance))
                      for observation in self.data]
        denominators = []
        for obs_index in range(self.n_obs):
            denominator = [np.exp(-np.linalg.norm(feature_obs_given - self.data[i, :][index_given])**2 / (2 * self.variance))
                           for i in range(self.n_obs)
                           if i != obs_index]
            denominators.append(sum(denominator))
        weights = [nominator / denominator for nominator, denominator in zip(nominators, denominators)]
        # sample index i  from weights distribution
        distribution_index = random.choices(list(range(self.n_obs)), weights=weights, k=1)[0]
        # sample from normal distribution i
        # get observation of distribution_index
        dist_obs = self.data[distribution_index, :]
        # extract only features to be imputed
        dist_obs_impute = dist_obs[index_impute]
        # sample from MVN with mean = dist_obs_impute und variance self.variance * I
        covariance = self.variance * np.identity(len(dist_obs_impute))
        sample = rng.multivariate_normal(mean=dist_obs_impute,cov=covariance, size=1)
        # fill up observation with imputed features
        feature_obs_arr[index_impute] = sample
        return feature_obs_arr
    

class EmpiricalRandomImputer:
    def __init__(self, data):
        self.data = data
        
    def fit(self):
        return self
    
    def predict(self, feature_observation, index_obs = None):
        feature_obs_arr = np.array(feature_observation, copy=True)
        # calculate weights
        # I first have to extract the indizes of given and to be imputed features
        index_given = np.where(feature_obs_arr != 0)[0]
        index_impute = np.where(feature_obs_arr == 0) # I assume every obs to be imputed is 0
        # delete columns of given features and row of current observation from original data
        data_impute = np.delete(np.delete(self.data, index_obs, axis=0), index_given, axis = 1)
        #data_impute = pd.DataFrame(np.delete(self.data, index_obs, axis=0)) # too bloated
        # sample random element from each remaining column
        n_features_to_impute = len(feature_obs_arr) - len(index_given)
        mask = list(np.random.randint(low=0, high=data_impute.shape[0], size=n_features_to_impute))
        indizes = [(mask_index, feature_index) for feature_index, mask_index in enumerate(mask)]
        imputed_features = data_impute[tuple(np.transpose(indizes))]
        #imputed_features = [np.random.choice(row) for row in data_impute.transpose()]
        #imputed_features = [np.random.choice(data_impute[column].values)
        #                    for column_index, column in enumerate(data_impute)
        #                    if column_index not in index_given]
        feature_obs_arr[index_impute] = imputed_features
        return feature_obs_arr
    
    
def get_imputer(imputer_name):
    return NormalCKDEImputer if imputer_name == "normal" else EmpiricalRandomImputer
    
    
        
    
        
        