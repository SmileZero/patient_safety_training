# How to do online training:

### Execute:

python OnlineTraining.py <True/False>

Argument is the validation indicator to see if we need to do the validation during this training.

### Example:

python OnlineTraining.py True

# How to extract AE tags:

### Execute:

python AEExtract.py startID endID

### Example:

python AEExtract.py 5000000 5000100

## File explanations:

### 1. current_cnn_model 

Folder stores the up to date model data including the meta file and checkpoint file.

### 2. last_cnn_model 

Folder stores the last version model data including the meta file and checkpoint file.

(Doing online training will automatically update the two folders)

(AEExtract file will use current cnn model as the model to do the prediction)

### 3. TrainingData.csv 

Stores all the training data that feed the current model

(Doing online training will automatically update this file to the completed version)

### 4. ValidationData.csv 

Stores all the validation data that randomly selected from the training data (1:9)

(Doing online training will automatically update this file to the completed version)
