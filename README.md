# Sleep Health and Lifestyle Data Analysis

This repository contains code for analyzing the "Sleep Health and Lifestyle" dataset. The dataset comprises various factors related to sleep health, lifestyle, and demographics. The analysis includes the distribution of gender and occupations among the participants, as well as two machine learning models: Multiple Linear Regression and Support Vector Classification (SVC).

## Dataset

The dataset used for this analysis is stored in a CSV file named `Sleep_health_and_lifestyle_dataset.csv`. The data is loaded using the pandas library to perform the analysis. The dataset contains the following columns:

- Gender
- Age
- Occupation
- BMI Category
- Sleep Duration
- Quality of Sleep
- Physical Activity Level
- Stress Level
- Heart Rate
- Daily Steps

## Demographics Analysis

The code provided in the notebook analyzes the demographics of the dataset, including the gender distribution and occupation distribution. It utilizes matplotlib for visualizations to understand the distribution of gender and occupations among the participants.
![Gender Distribution](https://github.com/egemengundur/Multiple_LInear_Regression_and_SVM_Algorithm/assets/75498353/42db417a-b90b-4f2f-8001-87be5ed83dc3)
![Occupation Distribution](https://github.com/egemengundur/Multiple_LInear_Regression_and_SVM_Algorithm/assets/75498353/ac96ac9b-764e-4145-97da-1baf2970a21a)
## Multiple Linear Regression Model

The first machine learning model applied to the data is Multiple Linear Regression. The code uses the `LinearRegression` class from scikit-learn to build the regression model. It predicts the "Quality of Sleep" based on several predictors: "Physical Activity Level," "Sleep Duration," "Stress Level," "Heart Rate," and "Age."

The following information is provided for the Multiple Linear Regression model:

- Coefficients:
    - B1: 0.01
    - B2: 0.62
    - B3: -0.27
    - B4: -0.05
    - B5: 0.01
- Intercept: 6.75
- The equation:
![image](https://github.com/egemengundur/Multiple_LInear_Regression_and_SVM_Algorithm/assets/75498353/d96aef48-9ff8-44c0-9177-ec6d8622f63d)

- R-squared: 0.91
- Standard errors of coefficients:
    - 0.001
    - 0.031
    - 0.021
    - 0.005
    - 0.002
- P-values:
    - 0.0245536511
    - 0.0000000000
    - 0.0000000000
    - 0.0000000000
    - 0.0000000036
- All the coefficients are significant at 0.05 significance level:

## Support Vector Classification (SVC) Model

The second machine learning model applied to the data is Support Vector Classification (SVC). The code uses the `SVC` class from scikit-learn to build the classification model. It predicts the "BMI Category" based on predictors: "Gender," "Age," "Sleep Duration," "Quality of Sleep," "Physical Activity Level," "Stress Level," "Heart Rate," and "Daily Steps."

The following information is provided for the SVC model:

- Confusion Matrix: A heatmap of the confusion matrix showing the performance of the model on the test data.
  ![Confusion Matrix](https://github.com/egemengundur/Multiple_LInear_Regression_and_SVM_Algorithm/assets/75498353/5bf6412b-71a4-40a4-a785-79a38c31d6fb)
- Accuracy: 0.96
- Precision: 0.96
- Recall: 0.96

## Usage

To run the analysis and the machine learning models, ensure you have the required libraries installed. The necessary libraries include pandas, numpy, matplotlib, and scikit-learn. You can install them using pip:

pip install pandas numpy matplotlib scikit-learn

Once the dependencies are installed, download the dataset (`Sleep_health_and_lifestyle_dataset.csv`) and place it in the same directory as the Jupyter notebook containing the code.

Open the Jupyter notebook and execute the code cells sequentially to perform the analysis and obtain the results of the machine learning models.

## License

The code in this repository is provided under the MIT License. Feel free to use and modify it as needed.

**Contact:**  
For any questions or issues, feel free to open an issue in this repository or contact Mehmet Egemen Gündür at gunduregemen@gmail.com 
