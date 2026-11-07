# Machine_Learning_Assignment2

# Dry Bean Classification

Live app: 

## a. Problem statement

Dry beans of different varieties look very similar, and sorting them by hand is slow and error-prone. This project compares five classification models that predict the variety of a dry bean from 16 shape and size measurements taken from images of the bean. The trained models are served through a Streamlit app, where any test CSV can be uploaded and scored with the saved models.

## b. Dataset description

- **Source:** UCI Machine Learning Repository, Dry Bean Dataset.
- **Size:** 13,611 rows in the original data. 68 exact duplicate rows were removed, which leaves 13,543 rows.
- **Features:** 16 numeric columns: Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness and ShapeFactor1 to ShapeFactor4. There are no missing values. The column name `AspectRation` is a typo in the original data and was kept as it is.
- **Target:** `Class`, with 7 bean types: Barbunya, Bombay, Cali, Dermason, Horoz, Seker and Sira. This is a multi-class problem. Dermason is the largest class (709 rows in the test set) and Bombay the smallest (104 rows).
- **Split:** 80/20, stratified by class, `random_state=42`. This gives 10,834 training rows and 2,709 test rows. `test_data.csv` is the test split.
- **Preprocessing:** the features are on very different scales (Area is in the tens of thousands, Solidity is close to 1). A `StandardScaler` is part of the pipeline for Logistic Regression, KNN, Naive Bayes and SVM, and it is saved inside each model file, so raw data can go straight into the saved models. The Decision Tree does not need scaling.

## c. GitHub repository

https://github.com/Kalpesh-18/Machine_Learning_Assignment2

Layout:

- `app.py`: Streamlit app
- `scripts/`: `config.py`, `prepare_data.py`, `evaluate.py`, one `train_*.py` per model, and `compare.py`
- `notebook/`: `01_eda.ipynb` (exploration) and `02_experiments.ipynb` (hyperparameter experiments; the paths in it are specific to the BITS Lab)
- `model/`: the five saved models and `results.csv`
- `data/`: the raw dataset and `train.csv`
- `test_data.csv`: the held-out test split

To run locally: `pip install -r requirements.txt`, then `streamlit run app.py`.

## d. Models used

All five models were trained on the same training split and scored on the same 2,709-row test set. Precision, recall and F1 are macro-averaged, so every class counts equally. AUC is one-vs-rest, macro-averaged. Hyperparameters were chosen with 5-fold cross-validation on the training data only, and the test set was used only for the final scores.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9192 | 0.9947 | 0.9307 | 0.93 | 0.9302 | 0.9023 |
| Decision Tree | 0.8944 | 0.9783 | 0.9075 | 0.907 | 0.9071 | 0.8724 |
| KNN | 0.9188 | 0.991 | 0.9328 | 0.9293 | 0.9306 | 0.9018 |
| Naive Bayes | 0.897 | 0.9911 | 0.9081 | 0.9082 | 0.9072 | 0.8762 |
| Support Vector Machine | 0.921 | 0.9945 | 0.9342 | 0.9317 | 0.9328 | 0.9045 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Does well for a linear model and has the highest AUC of the five (0.9947). In cross-validation, accuracy went from 0.9188 at C=0.01 to 0.9261 at C=100, but the gain above C=1 was about 0.002, which is within the fold-to-fold noise, so the default C=1 was kept. Scaling the features was needed for it to converge properly. |
| Decision Tree | Weakest model on accuracy (0.8944) and AUC (0.9783). With no depth limit it reached 1.0 training accuracy but only 0.897 in validation, so it overfits. Validation accuracy peaked at depth 8 to 9 (0.9079 and 0.9088, a difference too small to matter), so the shallower depth 8 was used. It relied mostly on ShapeFactor1, Compactness, MajorAxisLength and Perimeter. Its coarse leaf probabilities probably explain why its AUC is lower than the others. |
| KNN | Scaling made the biggest difference here: about 0.92 accuracy scaled against about 0.70 unscaled, because large-valued features like Area dominate the distance otherwise. k=1 and k=3 were worse, and k=11 to k=31 performed about the same, so k=21 was chosen. Test accuracy (0.9188) is almost identical to Logistic Regression. |
| Naive Bayes | With default settings it scored only about 0.76 in cross-validation, but standardising the features lifted it to about 0.90. After scaling, `var_smoothing` made no difference. Test accuracy was 0.8970, close to the Decision Tree, but its AUC (0.9911) is much higher, so it ranks the classes well even when its hard predictions are less accurate. Removing the near-duplicate size columns did not help, so the independence assumption did not hurt accuracy as much as expected. |
| Support Vector Machine | Best on accuracy, precision, recall, F1 and MCC. The rbf kernel beat the linear kernel slightly in cross-validation (0.9319 against 0.9271), and C values from 1 to 100 gave the same result, so C=1 was used. Per class, Bombay is classified perfectly (F1 1.000) and Sira is the hardest (F1 0.868). |
| **Overall winner** | **Support Vector Machine**, by a small margin. It leads on 5 of the 6 metrics (Logistic Regression is slightly ahead on AUC), but its accuracy lead over Logistic Regression and KNN is under 0.003, which is less than the noise expected on a 2,709-row test set. So SVM, Logistic Regression and KNN form a top group, and Naive Bayes and the Decision Tree are about two points below. |

### Other notes

- Most models scored a little lower on the test set than in cross-validation (for example the SVM went from about 0.932 to 0.921), so this particular test split may be slightly harder than the average validation fold.
- For Logistic Regression, Decision Trees, KNN, Naive Bayes and the SVM, the most confused pairs were Dermason/Sira and Barbunya/Cali, which suggests these bean types overlap in the measurements themselves.