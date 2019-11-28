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
