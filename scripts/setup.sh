#!bin/bash

curl -o RECORDS https://physionet.org/files/circor-heart-sound/1.0.3/RECORDS?download
curl -0 training_data.csv https://physionet.org/files/circor-heart-sound/1.0.3/training_data.csv?download
mkdir training_data