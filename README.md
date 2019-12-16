# ITU Research project Course 2019 - Spatial inforamation extraction toolkit.

### Training and testing the model

To train and test the model run:
> python main.py 

with one of the following parameters:
* -m : Model name. Defaults to blank 'en' model.
* -o :Optional output directory
* -nm :New model name for model meta. 
* -n : Number of training iterations

Default values of those are: (m=None, nm=SpRL, o=None, n= 30)

The default loaded dataset consist of the one stored in the folder dataset_1, which consist of IAPR TC-12 image benchmark corpus provided by SemEval. 
Showed accuracy consist of F1 score, and they are displayed showing the result of task 1: spatial entity recognition and task 2: spatial relation recognition, respectively.


### Testing a model with custom sentence

To test a model with a custom sentence:
> python script.py 

with one of the following parameters:
* -m : Model name. If no model is provided then the script won't be executed
* -s: Sentence which the model will extract the spatial information