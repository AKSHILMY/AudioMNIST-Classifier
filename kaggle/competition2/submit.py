#!/usr/bin/env python
# coding: utf-8

# ## Install & Imports

# In[1]:


get_ipython().run_line_magic('pip', 'install numpy pandas matplotlib seaborn scikit-learn')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Initialization of global variables

# In[3]:


LABEL_1 = "label_1"
LABEL_2 = "label_2"
LABEL_3 = "label_3"
LABEL_4 = "label_4"

NUM_OF_FEATURES = 768
LABELS = [LABEL_1,LABEL_2,LABEL_3,LABEL_4]
FEATURES = [f"feature_{i+1}" for i in range(0,NUM_OF_FEATURES)]


# ## Analysis of Dataset

# In[4]:


TRAIN_DF = pd.read_csv("dataset/train.csv")
VALID_DF = pd.read_csv("dataset/valid.csv")
TEST_DF = pd.read_csv("dataset/test.csv")


# In[5]:


assert len(TRAIN_DF.columns) - 4 == NUM_OF_FEATURES


# ## Preprocessing

# In[6]:


from sklearn.preprocessing import RobustScaler
from sklearn import svm
from sklearn import metrics


# In[7]:


x_train_dict = {}
y_train_dict = {}
x_valid_dict = {}
y_valid_dict = {}


# In[8]:


for target_label in LABELS:
  train_df_copy = TRAIN_DF[TRAIN_DF[LABEL_2].notna()] if target_label == LABEL_2 else TRAIN_DF
  valid_df_copy = VALID_DF[VALID_DF[LABEL_2].notna()] if target_label == LABEL_2 else VALID_DF

  scaler = RobustScaler()

  x_train_dict[target_label] = pd.DataFrame(scaler.fit_transform(train_df_copy.drop(LABELS,axis=1)),columns=FEATURES)
  y_train_dict[target_label] = train_df_copy[target_label]

  x_valid_dict[target_label] = pd.DataFrame(scaler.transform(valid_df_copy.drop(LABELS,axis=1)),columns=FEATURES)
  y_valid_dict[target_label] = valid_df_copy[target_label]


# ## Feature Engineering

# In[9]:


from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA


# In[10]:


def select_k_best_using_ANOVA_F(x_train,y_train,x_valid,k=100):
  selector = SelectKBest(f_classif,k = k)
  x_train_now = selector.fit_transform(x_train,y_train)
  x_valid_now = selector.transform(x_valid)
  return x_train_now,x_valid_now,selector

def PCA_transform(x_train,x_valid,n_components=0.95,svd_solver="full"):
  pca = PCA(n_components=n_components,svd_solver=svd_solver)
  pca.fit(x_train)
  x_train_trf = pd.DataFrame(pca.transform(x_train))
  x_valid_trf = pd.DataFrame(pca.transform(x_valid))
  return x_train_trf,x_valid_trf,pca

def get_accuracy(x_train,y_train,x_valid,y_valid,classifier="svc",params={"kernel" : "linear","average": "weighted","class_weight": None}):
  if classifier=="svc":
    classifier = svm.SVC(kernel=params['kernel'],class_weight = params["class_weight"])
    classifier.fit(x_train,y_train)
  y_pred = classifier.predict(x_valid)
  conf_matrix = metrics.confusion_matrix(y_valid,y_pred)
  accuracy = metrics.accuracy_score(y_valid,y_pred)
  precision = metrics.precision_score(y_valid,y_pred,average=params["average"])
  recall = metrics.recall_score(y_valid,y_pred,average=params["average"])
  return conf_matrix,accuracy,precision,recall


# ## Label 01 : Model Training, Validation & Testing

# In[11]:


conf_matrix_before,accuracy_before,precision_before,recall_before = get_accuracy(
    x_train= x_train_dict[LABEL_1],
    y_train = y_train_dict[LABEL_1],
    x_valid = x_valid_dict[LABEL_1],
    y_valid = y_valid_dict[LABEL_1],
    classifier="svc",
    params = {
        "kernel" : "linear",
        "average" : "weighted",
        "class_weight": None
    }
)


# In[12]:


print(f"Accuracy: {accuracy_before}")
print(f"Precision: {precision_before}")
print(f"Recall: {recall_before}")


# In[13]:


num_of_features_expected = 768
# Collection of all 768 features is not yet required
x_train_now,x_valid_now,selector = select_k_best_using_ANOVA_F(
    x_train = x_train_dict[LABEL_1],
    y_train = y_train_dict[LABEL_1],
    x_valid = x_valid_dict[LABEL_1],
    k = num_of_features_expected,
)
x_train_trf,x_valid_trf,pca = PCA_transform(
    x_train = x_train_now,
    x_valid = x_valid_now,
    n_components = 0.99,
    svd_solver = "full",

)
conf_matrix,accuracy,precision,recall = get_accuracy(
    x_train= x_train_trf,
    y_train = y_train_dict[LABEL_1],
    x_valid = x_valid_trf,
    y_valid = y_valid_dict[LABEL_1],
    classifier="svc",
    params = {
        "kernel" : "linear",
        "average" : "weighted",
        "class_weight": None
    }
)


# In[14]:


print(f"Number of features : {x_train_trf.columns}") 


# In[15]:


print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")


# In[16]:


TEST_DF = pd.read_csv("dataset/test.csv")
IDS = TEST_DF[TEST_DF.columns[0]]
features_df = TEST_DF[TEST_DF.columns[1:]]


# In[17]:


scaled_features_df = pd.DataFrame(scaler.fit_transform(features_df),columns=FEATURES)


# In[18]:


k = 768
n_components = 0.99
selector = SelectKBest(f_classif,k=k)
x_train_now = selector.fit_transform(x_train_dict[LABEL_1],y_train_dict[LABEL_1])
pca = PCA(n_components=n_components,svd_solver="full")
pca.fit(x_train_now)
x_train_trf = pd.DataFrame(pca.transform(x_train_now))

scaled_features_df_now = selector.transform(scaled_features_df)
scaled_features_df_now = pca.transform(scaled_features_df_now)

classifier = svm.SVC(kernel="linear",class_weight = None)
classifier.fit(x_train_trf,y_train_dict[LABEL_1])
labels_after = classifier.predict(scaled_features_df_now)


# In[19]:


assert len(features_df) == len(labels_after)


# In[20]:


submission = pd.DataFrame()
submission = pd.concat([submission,IDS],axis=1)


# In[21]:


submission = pd.concat([submission,pd.DataFrame(labels_after,columns=['label_1'])],ignore_index=False,axis=1)


# In[22]:


submission


# ## Label 02 : Model Training, Validation & Testing

# In[23]:


conf_matrix_before,accuracy_before,precision_before,recall_before = get_accuracy(
    x_train= x_train_dict[LABEL_2],
    y_train = y_train_dict[LABEL_2],
    x_valid = x_valid_dict[LABEL_2],
    y_valid = y_valid_dict[LABEL_2],
    classifier="svc",
    params = {
        "kernel" : "linear",
        "average" : "weighted",
        "class_weight": None
    }
)


# In[24]:


print(f"Accuracy: {accuracy_before}")
print(f"Precision: {precision_before}")
print(f"Recall: {recall_before}")


# In[25]:


num_of_features_expected = 768
# Collection of all 768 features is not yet required
x_train_now,x_valid_now,selector = select_k_best_using_ANOVA_F(
    x_train = x_train_dict[LABEL_2],
    y_train = y_train_dict[LABEL_2],
    x_valid = x_valid_dict[LABEL_2],
    k = num_of_features_expected,
)
x_train_trf,x_valid_trf,pca = PCA_transform(
    x_train = x_train_now,
    x_valid = x_valid_now,
    n_components = 0.99,
    svd_solver = "full",

)
conf_matrix,accuracy,precision,recall = get_accuracy(
    x_train= x_train_trf,
    y_train = y_train_dict[LABEL_2],
    x_valid = x_valid_trf,
    y_valid = y_valid_dict[LABEL_2],
    classifier="svc",
    params = {
        "kernel" : "linear",
        "average" : "weighted",
        "class_weight": None
    }
)


# In[26]:


print(f"Number of features : {x_train_trf.columns}") 


# In[27]:


print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")


# In[28]:


k = 768
n_components = 0.99
selector = SelectKBest(f_classif,k=k)
x_train_now = selector.fit_transform(x_train_dict[LABEL_2],y_train_dict[LABEL_2])
pca = PCA(n_components=n_components,svd_solver="full")
pca.fit(x_train_now)
x_train_trf = pd.DataFrame(pca.transform(x_train_now))

scaled_features_df_now = selector.transform(scaled_features_df)
scaled_features_df_now = pca.transform(scaled_features_df_now)

classifier = svm.SVC(kernel="linear",class_weight = None)
classifier.fit(x_train_trf,y_train_dict[LABEL_2])
labels_after = classifier.predict(scaled_features_df_now)


# In[29]:


assert len(features_df) == len(labels_after)


# In[30]:


submission = pd.concat([submission,pd.DataFrame(labels_after,columns=['label_2'])],ignore_index=False,axis=1)


# In[31]:


submission


# ## Label 03 : Model Training, Validation & Testing

# In[32]:


conf_matrix_before,accuracy_before,precision_before,recall_before = get_accuracy(
    x_train= x_train_dict[LABEL_3],
    y_train = y_train_dict[LABEL_3],
    x_valid = x_valid_dict[LABEL_3],
    y_valid = y_valid_dict[LABEL_3],
    classifier="svc",
    params = {
        "kernel" : "linear",
        "average" : "weighted",
        "class_weight": None
    }
)


# In[33]:


print(f"Accuracy: {accuracy_before}")
print(f"Precision: {precision_before}")
print(f"Recall: {recall_before}")


# In[34]:


num_of_features_expected = 768
# Collection of all 768 features is not yet required
x_train_now,x_valid_now,selector = select_k_best_using_ANOVA_F(
    x_train = x_train_dict[LABEL_3],
    y_train = y_train_dict[LABEL_3],
    x_valid = x_valid_dict[LABEL_3],
    k = num_of_features_expected,
)
x_train_trf,x_valid_trf,pca = PCA_transform(
    x_train = x_train_now,
    x_valid = x_valid_now,
    n_components = 0.99,
    svd_solver = "full",

)
conf_matrix,accuracy,precision,recall = get_accuracy(
    x_train= x_train_trf,
    y_train = y_train_dict[LABEL_3],
    x_valid = x_valid_trf,
    y_valid = y_valid_dict[LABEL_3],
    classifier="svc",
    params = {
        "kernel" : "linear",
        "average" : "weighted",
        "class_weight": None
    }
)


# In[35]:


print(f"Number of features : {x_train_trf.columns}") 


# In[36]:


print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")


# In[37]:


k = 768
n_components = 0.99
selector = SelectKBest(f_classif,k=k)
x_train_now = selector.fit_transform(x_train_dict[LABEL_3],y_train_dict[LABEL_3])
pca = PCA(n_components=n_components,svd_solver="full")
pca.fit(x_train_now)
x_train_trf = pd.DataFrame(pca.transform(x_train_now))

scaled_features_df_now = selector.transform(scaled_features_df)
scaled_features_df_now = pca.transform(scaled_features_df_now)

classifier = svm.SVC(kernel="linear",class_weight = None)
classifier.fit(x_train_trf,y_train_dict[LABEL_3])
labels_after = classifier.predict(scaled_features_df_now)


# In[38]:


assert len(features_df) == len(labels_after)


# In[39]:


submission = pd.concat([submission,pd.DataFrame(labels_after,columns=['label_3'])],ignore_index=False,axis=1)


# In[40]:


submission


# ## Label 04 : Model Training, Validation & Testing

# In[41]:


conf_matrix_before,accuracy_before,precision_before,recall_before = get_accuracy(
    x_train = x_train_dict[LABEL_4],
    y_train = y_train_dict[LABEL_4],
    x_valid = x_valid_dict[LABEL_4],
    y_valid = y_valid_dict[LABEL_4],
    classifier="svc",
    params = {
        "kernel" : "linear",
        "average" : "weighted",
        "class_weight": "balanced"
    }
)


# In[42]:


print(f"Accuracy: {accuracy_before}")
print(f"Precision: {precision_before}")
print(f"Recall: {recall_before}")


# In[43]:


num_of_features_expected = 768
# Collection of all 768 features is not yet required
x_train_now,x_valid_now,selector = select_k_best_using_ANOVA_F(
    x_train = x_train_dict[LABEL_4],
    y_train = y_train_dict[LABEL_4],
    x_valid = x_valid_dict[LABEL_4],
    k = num_of_features_expected,
)
x_train_trf,x_valid_trf,pca = PCA_transform(
    x_train = x_train_now,
    x_valid = x_valid_now,
    n_components = 0.99,
    svd_solver = "full",

)
conf_matrix,accuracy,precision,recall = get_accuracy(
    x_train= x_train_trf,
    y_train = y_train_dict[LABEL_4],
    x_valid = x_valid_trf,
    y_valid = y_valid_dict[LABEL_4],
    classifier="svc",
    params = {
        "kernel" : "linear",
        "average" : "weighted",
        "class_weight": "balanced"
    }
)


# In[44]:


print(f"Number of features : {x_train_trf.columns}") 


# In[45]:


print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")


# In[46]:


k = 768
n_components = 0.99
selector = SelectKBest(f_classif,k=k)
x_train_now = selector.fit_transform(x_train_dict[LABEL_4],y_train_dict[LABEL_4])
pca = PCA(n_components=n_components,svd_solver="full")
pca.fit(x_train_now)
x_train_trf = pd.DataFrame(pca.transform(x_train_now))

scaled_features_df_now = selector.transform(scaled_features_df)
scaled_features_df_now = pca.transform(scaled_features_df_now)

classifier = svm.SVC(kernel="linear",class_weight = "balanced")
classifier.fit(x_train_trf,y_train_dict[LABEL_4])
labels_after = classifier.predict(scaled_features_df_now)


# In[47]:


assert len(features_df) == len(labels_after)


# In[48]:


submission = pd.concat([submission,pd.DataFrame(labels_after,columns=['label_4'])],ignore_index=False,axis=1)


# In[49]:


submission


# ## Submission

# In[50]:


submission.to_csv("./submission.csv",index_label=False,index=False)

