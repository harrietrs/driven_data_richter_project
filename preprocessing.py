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

    
def train_test_split_function(train_x, train_y, stratify=False):
    if stratify:
        x_train,x_test,y_train,y_test=train_test_split(train_x,train_y,stratify=train_y,test_size=0.2)
        x_train,x_cv,y_train,y_cv=train_test_split(x_train,y_train,stratify=y_train,test_size=0.2)
    else:
        x_train,x_test,y_train,y_test=train_test_split(train_x,train_y,stratify=None,test_size=0.2)
        x_train,x_cv,y_train,y_cv=train_test_split(x_train,y_train,stratify=y_train,test_size=0.2)
    return x_train, x_test, x_cv, y_train, y_test, y_cv


def normalize_numerical_data(train_data,cv_data,test_data, normalizer=Normalizer()):
    std=normalizer
    std.fit(train_data)
    transformed_input=std.transform(train_data)
    transformed_cv=std.transform(cv_data)
    transformed_test=std.transform(test_data)
    return transformed_input,transformed_cv,transformed_test


def one_hot_encoding_categorical_data(train_data,cv_data,test_data, std_encoding=OneHotEncoder()):
    std=std_encoding
    std.fit(train_data)
    std_train=std.transform(train_data)
    std_test=std.transform(test_data)
    std_cv=std.transform(cv_data)
    
    encoder_labels = dict(zip(list(train_data.columns), [category.tolist() for category in std.categories_]))
    col_name_list = []
    for key, value in encoder_labels.items():
        for label in value:
            col_name_list.append(key + '_' + label)
        
    train_data= pd.DataFrame(data=std_train.todense(), columns=col_name_list)
    test_data = pd.DataFrame(data=std_test.todense(), columns=col_name_list)
    cv_data = pd.DataFrame(data=std_cv.todense(), columns=col_name_list)

    return train_data,cv_data,test_data


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
