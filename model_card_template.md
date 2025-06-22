# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This project trains a machine learning model to predict income levels using census data. 
It looks to predict if a person's income exceeds $50k/year.
The model is implemented using a scikit-learn classifier (Random Forest). The model was
trained on preprocessed census data and included categorical and continuous features.

## Intended Use
The model is intended to be used for demographic and socioeconomic research, providing predictions 
relating to income based on census data attributes. It can be used to analyze income distributions and 
identify key factors influencing income.

## Training Data
The training data used consists of the US census income dataset. This set includes demgraphic
and employment information for individuals. It was split into training and subsets for model development.

## Evaluation Data
The model was evaluated on a held-out test set derived from the same Census income dataset.
This test was processed using the same transformations as the training data for consistency.


## Metrics
The model's performance was evaluated using precision, recall, and F1 score metrics:
Precision: 0.7419
Recall: 0.6384
F1 Score: 0.6863

These metrics indicate the effectiveness in correctly predicting if an individual's 
income exceeds $50k while balancing false positives and false negatives.

## Ethical Considerations
The census data includes attributes such as race, sex, and native country. While these features help
improve accuracy, it needs to be ensured that the model does not reinforce social bias.
Users should evaluate fairness and potential disparate impacts before deployment.

## Caveats and Recommendations
The model's predictions should not be used as a sole decision maker in high-stakes settings.
The accuracy depends on the quality and representation of the census data.
For deployment in different regions or populations, the model should be retrained and validated.
Continuous monitoring is recommended to maintain ethical use.
