# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
I use Gradient Boosting Classifier to predict the salary.
## Intended Use
Use this model to predict the salary of a person depend on his financial data.
## Training Data
Data is coming from https://archive.ics.uci.edu/ml/datasets/census+income ; training data is done using 80% of this data.
## Evaluation Data
Data is coming from https://archive.ics.uci.edu/ml/datasets/census+income ; evaluation data is done using 20% of this data.
## Metrics
This model got avarage about Precision: 0.73,Recall: 0.57,Fbeta: 0.64.
## Ethical Considerations
Dataset contains race,sex,education and origin country that may make the model potentially discriminate people.
## Caveats and Recommendations
This model is tested with all feature in dataset.Please do feature selection and try with another model(e.g: Decision Tree)