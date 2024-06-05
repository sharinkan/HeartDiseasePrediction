# Deep Learning: Heart Disease Prediction

### TeamName: CardioWave Forecasters
### Instructors: Milad Toutounchian, Mahmoud Zeydabadinezhad
### Group members: Ziao You, Weijie Chen, Xiao Fang, Yixuan Song,  Kan Kim
### Email Address: zy364@drexel.edu



#

# Introduction

In the field of healthcare, the stethoscope has been an indispensable tool for diagnosing cardiovascular conditions by listening to the heart's sounds. These auditory cues provide valuable information to skilled medical professionals and we are attempting to use machine learning to give an early indication of heart conditions

# Methods

- Classical ML models such as KNN and SVM 
- Ensemble methods (voting)
- Mixture model with different features sets
- (Currently) Deep Learning models -> (CNN, MLP, combined MLP)

# Results Updated by 06/05/2024
| Model             | Features                                                 | Accuracy |
|-------------------|----------------------------------------------------------|----------|
| Single MLP        | Chromagram + 1D Mel-Spec + MFCC + Opensmile + CSV + Time Series | 66.46    |
| Multiscale 2D CNN | 2D Mel-Spec                                              | 58.23    |
| 1D CNN            | Bandpower Struct + CSV                                   | 68.95    |
| Concat-CNN        | Chromagram + 1D Mel-Spec + MFCC + Bandpower Struct + CSV | 64.22    |
| Conv-RNN          | 1D Mel-Spec + MFCC + Opensmile + CSV                     | 64.38    |
| ResNet            | Chromagram + 1D Mel-Spec + MFCC + Time Series            | 55.22    |
| Concat-MLP        | 2D Mel-Spec + Bandpower Struct + Opensmile + CSV         | 59.11    |
| LSTM              | Chromagram + 2D Mel-Spec + CSV + Time Series             | 53.85    |
| RNN               | Chromagram + Bandpower Struct + CSV + Time Series        | 50.00    |



# Results (ML)

### Different Input Features + Same ML Models + Voting
```
['mfcc'] + hard voting
	               Model  Accuracy   AUC
0      LogisticRegression  0.633987  0.64
1                     SVC  0.627451  0.63
2    KNeighborsClassifier  0.568627  0.57
3  DecisionTreeClassifier  0.526144  0.52
4  RandomForestClassifier  0.584967  0.59
5              GaussianNB  0.594771  0.60
ensemble_methods : VOTING='hard' 0.630718954248366

['mfcc'] + soft voting
                   Model  Accuracy   AUC
0      LogisticRegression  0.545752  0.55
1                     SVC  0.581699  0.59
2    KNeighborsClassifier  0.526144  0.53
3  DecisionTreeClassifier  0.555556  0.56
4  RandomForestClassifier  0.558824  0.57
5              GaussianNB  0.542484  0.54
ensemble_methods : VOTING='soft' 0.545751633986928


['feature_chromagram'] + hard voting
                   Model  Accuracy   AUC
0      LogisticRegression  0.578431  0.58
1                     SVC  0.555556  0.56
2    KNeighborsClassifier  0.549020  0.55
3  DecisionTreeClassifier  0.496732  0.50
4  RandomForestClassifier  0.535948  0.54
5              GaussianNB  0.529412  0.52
ensemble_methods : VOTING='hard' 0.5686274509803921

['feature_chromagram'] + soft voting


                   Model  Accuracy   AUC
0      LogisticRegression  0.532680  0.53
1                     SVC  0.526144  0.53
2    KNeighborsClassifier  0.509804  0.51
3  DecisionTreeClassifier  0.558824  0.56
4  RandomForestClassifier  0.513072  0.51
5              GaussianNB  0.500000  0.50
ensemble_methods : VOTING='soft' 0.5294117647058824


['feature_melspectrogram'] + hard voting
                   Model  Accuracy   AUC
0      LogisticRegression  0.575163  0.56
1                     SVC  0.555556  0.54
2    KNeighborsClassifier  0.575163  0.57
3  DecisionTreeClassifier  0.522876  0.52
4  RandomForestClassifier  0.571895  0.57
5              GaussianNB  0.539216  0.52
ensemble_methods : VOTING='hard' 0.565359477124183

['feature_melspectrogram'] + soft voting

                   Model  Accuracy   AUC
0      LogisticRegression  0.552288  0.52
1                     SVC  0.549020  0.50
2    KNeighborsClassifier  0.526144  0.53
3  DecisionTreeClassifier  0.509804  0.50
4  RandomForestClassifier  0.584967  0.59
5              GaussianNB  0.568627  0.51
ensemble_methods : VOTING='soft' 0.5620915032679739


[feature_bandpower]
                   Model  Accuracy   AUC
0      LogisticRegression  0.496732  0.51
1                     SVC  0.516340  0.53
2    KNeighborsClassifier  0.532680  0.53
3  DecisionTreeClassifier  0.519608  0.52
4  RandomForestClassifier  0.539216  0.54
5              GaussianNB  0.486928  0.52
ensemble_methods : VOTING='hard' 0.5065359477124183


[NMF]
                   Model  Accuracy   AUC
0      LogisticRegression  0.493464  0.49
1                     SVC  0.562092  0.56
2    KNeighborsClassifier  0.500000  0.50
3  DecisionTreeClassifier  0.480392  0.48
4  RandomForestClassifier  0.519608  0.52
5              GaussianNB  0.535948  0.51
ensemble_methods : VOTING='hard' 0.5392156862745098


['feature_mfcc', 'feature_chromagram']

                   Model  Accuracy   AUC
0      LogisticRegression  0.594771  0.60
1                     SVC  0.584967  0.59
2    KNeighborsClassifier  0.496732  0.50
3  DecisionTreeClassifier  0.539216  0.54
4  RandomForestClassifier  0.535948  0.54
5              GaussianNB  0.562092  0.56
ensemble_methods : VOTING='hard' 0.5686274509803921


[feature_mfcc, feature_bandpower]

                   Model  Accuracy   AUC
0      LogisticRegression  0.565359  0.57
1                     SVC  0.601307  0.60
2    KNeighborsClassifier  0.496732  0.49
3  DecisionTreeClassifier  0.555556  0.56
4  RandomForestClassifier  0.549020  0.55
5              GaussianNB  0.503268  0.51
ensemble_methods : VOTING='hard' 0.5751633986928104
```

### Different Input Features + Different ML Models 
```
Voting (hard)
                        Model  Accuracy   AUC
0                 LogisticRegression()_f1  0.594771  0.59
1                                SVC()_f1  0.591503  0.59
2                         GaussianNB()_f1  0.581699  0.58
3             RandomForestClassifier()_f1  0.539216  0.54
4                                SVC()_f2  0.486928  0.49
5  KNeighborsClassifier(n_neighbors=7)_f3  0.539216  0.54
6                 LogisticRegression()_f3  0.549020  0.55
ensemble_methods : VOTING='hard' 0.6111111111111112


Voting (soft)
                        Model  Accuracy   AUC
0      LogisticRegression()_f1  0.614379  0.61
1                     SVC()_f1  0.617647  0.62
2              GaussianNB()_f1  0.565359  0.57
3  RandomForestClassifier()_f1  0.604575  0.60
4      LogisticRegression()_f2  0.490196  0.49
5      LogisticRegression()_f3  0.549020  0.55
ensemble_methods : VOTING='soft' 0.5490196078431373

```



#
Google Colab Folder: https://drive.google.com/drive/folders/1R7UYthWO2DxDS4M7F3t_E6CDGoRBOWn4?usp=share_link


#
Dataset link: https://physionet.org/content/circor-heart-sound/1.0.3/

# 

## Setup script instruction
#### Python version : 3.9+
#

### Install required packages
`pip install -r requirements.txt`
#

### Install dataset
`python install.py`
- downloading from https://physionet.org/static/published-projects/circor-heart-sound/the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip

### Training
`python run.py`


