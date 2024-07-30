import pandas as pd
from pycaret.classification import *

# Load the dataset
data = pd.read_csv("D:/MicroGrid/sample_model/V2G_G2V.csv")

# Initialize the setup
setup(data=data, target='response', session_id=123)

# Compare models and select the best model
best = compare_models()

# Create and evaluate a decision tree model
dt_model = create_model('dt', criterion='gini')
evaluate_model(dt_model)

# Save the best model
save_model(best, 'best_model')
