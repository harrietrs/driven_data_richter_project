import pandas as pd
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
    