
# Feel free to add your own lines of code to handle further data manipulation (wrangling)
# before feeding it to future ML models
import pandas as pd



def data_wrangling(df : pd.DataFrame):

    data = df.copy(deep=True)
    data['Age'].fillna('Young Adult', inplace = True) # Labeling 'nan' values in age column

    # New columns
    data['Recording Channels'] = data['Recording locations:'].str.split('+')
    data['Murmur Channels'] = data['Murmur locations'].str.split('+')

    # Mapping of string values to integer values
    outcome_mapping = {'Normal': 0, 'Abnormal': 1}
    data['Outcome'] = data['Outcome'].replace(outcome_mapping)


    # Further mapping if required later
    # smt_mapping = {'Early-systolic': 0, 'Mid-systolic': 1, 'Late-systolic': 2, 'Holosystolic': 3}
    # sms_mapping = {'Crescendo': 0, 'Decrescendo': 1, 'Diamond': 2, 'Plateau': 3}
    # smg_mapping = {'I/VI': 0, 'II/VI': 1, 'III/VI': 2}
    # smp_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    # smq_mapping = {'Musical': 0, 'Blowing': 1, 'Harsh': 2}

    # dmt_mapping = {'Early-diastolic': 0, 'Holodiastolic': 1, 'Mid-diastolic': 2}
    # dms_mapping = {'Decrescendo': 0, 'Plateau': 1}
    # dmg_mapping = {'I/IV': 0, 'II/IV': 1, 'III/IV': 2}
    # dmp_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    # dmq_mapping = {'Blowing': 0, 'Harsh': 1}


    # Add new lines as needed for future work
    '''

    '''
    return data


def label_encoding(df: pd.DataFrame, cols: list):
    data = df.copy(deep=True)
    data = data[cols]
    for col in cols:
        data[col] = LabelEncoder().fit_transform(data[col])
        ## Figure out how to extract dict key-val matching pair
        # label_mapping = {encoded: label for encoded, label in enumerate(label_encoder.classes_)}

    # print(label_mapping)
    return data


def one_hot_encoding(df : pd.DataFrame, cols: list):
    data = df.copy(deep=True)
    data = data[cols]
    df_encoded = pd.get_dummies(data, dtype=int)

    return df_encoded