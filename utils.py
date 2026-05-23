import numpy as np
import pandas as pd

""" Standardisation """
class Standardisation():

    def fit(self, x: 'ndarray'):
        self.mean_x = x.mean(axis= 0)
        self.sd_x = x.std(axis= 0)

    def fit_transform(self, x: 'ndarray'):
        self.mean_x = x.mean(axis= 0)
        self.sd_x = x.std(axis= 0)
        x_new = np.copy(x)
        x_new = (x_new - self.mean_x) / self.sd_x
        return x_new

    def transform(self, x: 'ndarray'):
        x_new = np.copy(x)
        x_new = (x_new - self.mean_x) / self.sd_x
        return x_new

""" Pseudovariables """
class Dummies():

    def fit(self, df, columns= [], top_n_categories= None):
        df_copy = df.copy()
        self.columns = columns
        self.col_cat = {}
        for col in self.columns:
            if top_n_categories == None:
                self.col_cat[col] = df_copy[col].unique()
            else:
                self.col_cat[col] = df_copy[col].unique()[: top_n_categories + 1]

    def transform(self, df, drop= True, form= ''):
        df_copy = df.copy()
        self.columns_list = []
        for col in self.col_cat:
            if drop == True:
                for cat in self.col_cat[col][: -1]:
                    df_copy[f'{col}_{cat}'] = (df_copy[col] == cat).astype('int')
                    self.columns_list.append(f'{col}_{cat}')
            else:
                for cat in self.col_cat[col]:
                    df_copy[f'{col}_{cat}'] = (df_copy[col] == cat).astype('int')
                    self.columns_list.append(f'{col}_{cat}')
        if form == 'matrix':
            return df_copy[self.columns_list].values
        else:
            return df_copy