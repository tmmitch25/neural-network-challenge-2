# Neural Network Challenge 2

## Background

In this project, I developed a branched neural network for HR to predict two key outcomes for employees:
- **Attrition:** Whether an employee is likely to leave the company.
- **Department Fit:** Which department an employee may be best suited for.

The goal was to provide HR with actionable insights to both retain talent and optimize departmental placement.

---

## Files

- `attrition.ipynb`: Main Jupyter notebook containing all code, analysis, and results.

---

## Project Workflow

### Before You Begin

1. I created a new repository called `neural-network-challenge-2` and cloned it to my computer.
2. I added the starter file `attrition.ipynb` to the repository and pushed the initial commit to GitHub.
3. I worked on the notebook in Google Colab, periodically downloading and pushing updates to GitHub to keep my work backed up and versioned.

---

## Instructions & Approach

### Part 1: Preprocessing

- **Data Import:** Loaded the dataset and displayed the first five rows to understand the structure.
- **Exploration:** Determined the number of unique values in each column to identify categorical and numerical features.
- **Target Preparation:** Created `y_df` with the `Attrition` and `Department` columns as targets.
- **Feature Selection:** Chose at least 10 columns (excluding `Attrition` and `Department`) for the feature set `X_df`.
- **Data Types:** Displayed data types for all selected features.
- **Train/Test Split:** Split the data into training and testing sets.
- **Encoding:** Converted all categorical features to numeric using appropriate encoders (e.g., `pd.get_dummies`, `OneHotEncoder`).
- **Scaling:** Applied `StandardScaler` to the features.
- **Target Encoding:** One-hot encoded both the `Department` and `Attrition` columns for use in the neural network.

### Part 2: Model Creation, Compilation, and Training

- **Input Layer:** Determined the number of input features and created the input layer (using the functional API, not Sequential).
- **Shared Layers:** Built at least two shared hidden layers to learn common representations.
- **Branched Outputs:**
  - **Department Branch:** Added a hidden layer and an output layer with softmax activation for multi-class classification.
  - **Attrition Branch:** Added a hidden layer and an output layer with sigmoid activation for binary classification.
- **Model Compilation:** Compiled the model with appropriate loss functions (`categorical_crossentropy` for department, `binary_crossentropy` for attrition) and metrics.
- **Model Summary:** Displayed the model architecture.
- **Training:** Trained the model on the preprocessed data, monitoring accuracy for both outputs.
- **Evaluation:** Evaluated the model on the test set and printed accuracy for both department and attrition predictions.

### Part 3: Summary & Reflection

#### 1. Is accuracy the best metric to use on this data? Why or why not?

Accuracy is not always the best metric for attrition prediction because the dataset is often imbalanced (more "No" than "Yes" for attrition). In such cases, a model can achieve high accuracy by always predicting the majority class, but this does not mean it is good at identifying employees who actually leave. Better metrics for imbalanced data include precision, recall, F1-score, or ROC-AUC, as these provide more insight into the model’s ability to detect the minority class. For the Department output, if the classes are balanced, accuracy is more appropriate, but if not, similar caution applies.

#### 2. What activation functions did you choose for your output layers, and why?

- **Attrition output:** I used a sigmoid activation function for the Attrition output layer because this is a binary classification problem ("Yes" or "No"). Sigmoid outputs a probability between 0 and 1, which is ideal for binary outcomes.
- **Department output:** I used a softmax activation function for the Department output layer because this is a multi-class classification problem (predicting one of several departments). Softmax outputs a probability distribution across all possible classes, ensuring the probabilities sum to 1.

#### 3. Can you name a few ways that this model might be improved?

- **Address class imbalance:** Use class weighting, oversampling (e.g., SMOTE), or undersampling to help the model learn from rare events like attrition.
- **Tune hyperparameters:** Experiment with the number of layers, units per layer, learning rate, batch size, and activation functions.
- **Feature engineering:** Create new features, select the most relevant features, or use domain knowledge to improve input data.
- **Regularization:** Add dropout layers or L1/L2 regularization to reduce overfitting.
- **Model evaluation:** Use cross-validation and additional metrics (precision, recall, F1-score, ROC-AUC) to better assess model performance.
- **Ensemble methods:** Combine predictions from multiple models for improved accuracy and robustness.

---

## Requirements Checklist

- [x] Data imported and explored
- [x] Targets and features prepared
- [x] Data split, encoded, and scaled
- [x] Model created, compiled, and trained with branched outputs
- [x] Model evaluated and summary questions answered

---

## Author

Trayshawn Mitchell

