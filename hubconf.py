# kali
import torch
from torch import nn
import torch.optim as optim

# You can import whatever standard packages are required

# full sklearn, full pytorch, pandas, matplotlib, numpy are all available
# Ideally you do not need to pip install any other packages!
# Avoid pip install requirement on the evaluation program side, if you use above packages and sub-packages of them, then that is fine!

###### PART 1 ######-----------------------------------------------------------------------------------

from sklearn.datasets import make_blobs, make_circles
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_completeness_v_measure
def get_data_blobs(n_points=100):
  # write your code here
  # Refer to sklearn data sets
  
  X, y = make_blobs(n_samples=n_points, centers=3, n_features=2,random_state=0)
  return X,y

def get_data_circles(n_points=100):
  # write your code here
  # Refer to sklearn data sets
  X, y = make_circles(n_samples=n_points, random_state=0, factor=0.8)
  return X,y

def build_kmeans(X=None,k=10):
  # k is a variable, calling function can give a different number
  # Refer to sklearn KMeans method
  km = KMeans(n_clusters=k, random_state=0).fit(X)
  return km

def assign_kmeans(km=None,X=None):
  # For each of the points in X, assign one of the means
  # refer to predict() function of the KMeans in sklearn
  ypred = km.predict(X)
  return ypred

def compare_clusterings(ypred_1=None,ypred_2=None):
  # refer to sklearn documentation for homogeneity, completeness and vscore
  h,c,v = 0,0,0 # you need to write your code to find proper values
  h,c,v = homogeneity_completeness_v_measure(ypred_1, ypred_2, beta=1.0)
  return h,c,v

###### PART 2 ######-------------------------------------------------------------------------------------

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

def get_data_mnist():
  # Refer to sklearn data sets
  X,y = load_digits(return_X_y=True)
  return X,y

def build_lr_model(X=None, y=None):
  lr_model = LogisticRegression()
  lr_model.fit(X,y)
  # Build logistic regression, refer to sklearn
  return lr_model

def build_rf_model(X=None, y=None):
  rf_model = RandomForestClassifier(random_state=0)
  rf_model.fit(X,y)
  # Build Random Forest classifier, refer to sklearn
  return rf_model

def get_metrics(model=None,X=None,y=None):
  # Obtain accuracy, precision, recall, f1score, auc score - refer to sklearn metrics
  y_true = y
  y_pred = model.predict(X)
  acc, prec, rec, f1, auc = 0,0,0,0,0
  acc = accuracy_score(y_true, y_pred)
  prec = precision_score(y_true, y_pred, average='macro')
  rec = recall_score(y_true, y_pred, average='macro')
  f1 = f1_score(y_true, y_pred, average='macro')
  y_pred = model.predict_proba(X)
  # y_pred = np.transpose([pred[:, 1] for pred in y_pred])
  auc = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')
  return acc, prec, rec, f1, auc

def get_paramgrid_lr():
  # you need to return parameter grid dictionary for use in grid search cv
  # penalty: l1 or l2
  lr_param_grid = {'penalty':['l2', None]}
  # refer to sklearn documentation on grid search and logistic regression
  return lr_param_grid

def get_paramgrid_rf():
  # you need to return parameter grid dictionary for use in grid search cv
  # n_estimators: 1, 10, 100
  # criterion: gini, entropy
  # maximum depth: 1, 10, None  
  rf_param_grid = {'n_estimators':[1,10,100],'max_depth':[1,10, None], 'criterion':['gini', 'entropy']}
  # refer to sklearn documentation on grid search and random forest classifier
  return rf_param_grid

def perform_gridsearch_cv_multimetric(model=None, param_grid=None, cv=5, X=None, y=None, metrics=['accuracy','roc_auc_ovr']):
  # you need to invoke sklearn grid search cv function
  # refer to sklearn documentation
  # the cv parameter can change, ie number of folds  
  
  # metrics = [] the evaluation program can change what metrics to choose

  # This check is there because roc_auc does not work for multi class. We have to use roc_auc_ovr
  for i in range (0, len(metrics)):
    if metrics[i] == 'roc_auc':
      metrics[i] ='roc_auc_ovr'

  # print(metrics)
  
  grid_search_cv = GridSearchCV(estimator = model, param_grid = param_grid, scoring = metrics, refit = 'roc_auc_ovr')
  grid_search_cv.fit(X,y)
  
  # metric of choice will be asked here, refer to the-scoring-parameter-defining-model-evaluation-rules of sklearn documentation
  
  # refer to cv_results_ dictonary
  # return top 1 score for each of the metrics given, in the order given in metrics=... list
  data = grid_search_cv.cv_results_
  # print(data)
  
  top1_scores = []
  top1_scores.append(grid_search_cv.best_estimator_)
  top1_scores.append(grid_search_cv.best_score_)
  
  return top1_scores

# ###### PART 3 ######-------------------------------------------------------------------

# class MyNN(nn.Module):
#   def __init__(self,inp_dim=64,hid_dim=13,num_classes=10):
#     super(MyNN,self)
    
#     self.fc_encoder = None # write your code inp_dim to hid_dim mapper
#     self.fc_decoder = None # write your code hid_dim to inp_dim mapper
#     self.fc_classifier = None # write your code to map hid_dim to num_classes
    
#     self.relu = None #write your code - relu object
#     self.softmax = None #write your code - softmax object
    
#   def forward(self,x):
#     x = None # write your code - flatten x
#     x_enc = self.fc_encoder(x)
#     x_enc = self.relu(x_enc)
    
#     y_pred = self.fc_classifier(x_enc)
#     y_pred = self.softmax(y_pred)
    
#     x_dec = self.fc_decoder(x_enc)
    
#     return y_pred, x_dec
  
#   # This a multi component loss function - lc1 for class prediction loss and lc2 for auto-encoding loss
#   def loss_fn(self,x,yground,y_pred,xencdec):
    
#     # class prediction loss
#     # yground needs to be one hot encoded - write your code
#     lc1 = None # write your code for cross entropy between yground and y_pred, advised to use torch.mean()
    
#     # auto encoding loss
#     lc2 = torch.mean((x - xencdec)**2)
    
#     lval = lc1 + lc2
    
#     return lval
    
# def get_mynn(inp_dim=64,hid_dim=13,num_classes=10):
#   mynn = MyNN(inp_dim,hid_dim,num_classes)
#   mynn.double()
#   return mynn

# def get_mnist_tensor():
#   # download sklearn mnist
#   # convert to tensor
#   X, y = None, None
#   # write your code
#   return X,y

# def get_loss_on_single_point(mynn=None,x0,y0):
#   y_pred, xencdec = mynn(x0)
#   lossval = mynn.loss_fn(x0,y0,y_pred,xencdec)
#   # the lossval should have grad_fn attribute set
#   return lossval

# def train_combined_encdec_predictor(mynn=None,X,y, epochs=11):
#   # X, y are provided as tensor
#   # perform training on the entire data set (no batches etc.)
#   # for each epoch, update weights
  
#   optimizer = optim.SGD(mynn.parameters(), lr=0.01)
  
#   for i in range(epochs):
#     optimizer.zero_grad()
#     ypred, Xencdec = mynn(X)
#     lval = mynn.loss_fn(X,y,ypred,Xencdec)
#     lval.backward()
#     optimzer.step()
    
#   return mynn