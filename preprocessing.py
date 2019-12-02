import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, OneHotEncoder


def resample(train_data, sample_size=25000):
    # resampling
    train_high=train_data[train_data.damage_grade==3]
    train_medium=train_data[train_data.damage_grade==2]
    train_low=train_data[train_data.damage_grade==1]
    
    train_high=train_high.sample(
        sample_size, 
#         resample=True,
        random_state=42)
    train_medium=train_medium.sample(
        sample_size, 
#         resample=True,
        random_state=42)
    train_low=train_low.sample(
        sample_size, 
#         resample=True,
        random_state=42)
   
    train = pd.concat([train_high, train_medium,train_low], ignore_index=True, sort =False)
    
    return train


def get_new_features(train_data, test_data):
    
    train_data = make_new_features(train_data)
    test_data = make_new_features(test_data)
    
    return train_data, test_data
    
    
    

def make_new_features(data):
    
    data['floors_per_area'] = (data.count_floors_pre_eq/data.area_percentage).copy()
    data['floors_per_height'] = (data.count_floors_pre_eq/data.height_percentage).copy()
    data['families_per_floor'] = (data.count_families/data.count_floors_pre_eq).copy()
    data['families_per_area'] = (data.count_families/data.area_percentage).copy()
    data['families_per_height'] = (data.count_families/data.height_percentage).copy()
    
    return data

    
def train_test_split_function(train_x, train_y, split=0.2, stratify=False):
    if stratify:
        x_train,x_test,y_train,y_test=train_test_split(train_x,train_y,stratify=train_y,test_size=split)
        x_train,x_cv,y_train,y_cv=train_test_split(x_train,y_train,stratify=y_train,test_size=split)
    else:
        x_train,x_test,y_train,y_test=train_test_split(train_x,train_y,stratify=None,test_size=split)
        x_train,x_cv,y_train,y_cv=train_test_split(x_train,y_train,stratify=y_train,test_size=split)

    return x_train, x_test, x_cv, y_train, y_test, y_cv


def normalize_numerical_data(train_data,cv_data,test_data, unlabelled_data, normalizer=Normalizer()):
    std=normalizer
    std.fit(train_data)
    transformed_input=std.transform(train_data)
    transformed_cv=std.transform(cv_data)
    transformed_test=std.transform(test_data)
    transformed_unlabelled_data=std.transform(unlabelled_data)
    return transformed_input,transformed_cv,transformed_test, transformed_unlabelled_data


def one_hot_encoding_categorical_data(train_data,cv_data,test_data, unlabelled_data, std_encoding=OneHotEncoder()):
    train_data = train_data.astype('category')
    cv_data = cv_data.astype('category')
    test_data = test_data.astype('category')
    unlabelled_data = unlabelled_data.astype('category')
    
    train_data = pd.get_dummies(train_data)
    cv_data = pd.get_dummies(cv_data)
    test_data = pd.get_dummies(test_data)
    unlabelled_data = pd.get_dummies(unlabelled_data)
    
    final_train, final_unlabelled = train_data.align(unlabelled_data,
                                                                    join='left', 
                                                                    axis=1)
    final_unlabelled = final_unlabelled.fillna(0)
    
    final_train, final_test = final_train.align(test_data,
                                                                    join='left', 
                                                                    axis=1)
    
    final_test = final_test.fillna(0)
    final_train, final_cv = final_train.align(cv_data,
                                                                    join='left', 
                                                                    axis=1)
    final_cv = final_cv.fillna(0)

    
    return final_train,final_cv,final_test, final_unlabelled


def add_polynomials(data, degree=2, feats_for_poly = ['age','area_percentage','height_percentage','count_floors_pre_eq','count_families',
                    'floors_per_area', 'floors_per_height', 'families_per_floor', 'families_per_area',
                    'families_per_height']):
    
    
    poly = PolynomialFeatures(degree)
    polynomial_feats = poly.fit_transform(data[feats_for_poly])
    polynomial_features_df = pd.DataFrame(data=polynomial_feats,columns=poly.get_feature_names(feats_for_poly),dtype='int64')
    polynomial_features_df.drop('1',axis=1,inplace=True)
    polynomial_features_df.drop(feats_for_poly,axis=1,inplace=True)
    
    data_with_polynomial_features = pd.concat([data, polynomial_features_df], axis=1)
    
    return data_with_polynomial_features

def drop_sparse_features(data, threshold=0.9999):
    num_zeros = (((data == 0).sum())/data.shape[0]).sort_values(ascending=False)
    zero_cols_to_drop = list(num_zeros[num_zeros>threshold].index)
    print("{} columns dropped".format(str(len(ero_cols_to_drop))))
    trimmed_data = data.drop(columns=zero_cols_to_drop)
    return trimmed_data
