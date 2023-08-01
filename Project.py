import pandas as pd
import numpy as np

data = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
data.drop(['Person ID'], axis=1, inplace=True)


#### DEMOGRAPHICS
import matplotlib.pyplot as plt
data['Occupation'] = data['Occupation'].str.replace('Sales Representative', 'Sales Rep.')
data['Occupation'] = data['Occupation'].str.replace('Software Engineer', 'SW Engineer')

# Gender Distribution
data.Gender= pd.Series(data.Gender)
gender_counts = data.Gender.value_counts()
categories = ['Male', 'Female']

plt.style.use("ggplot")
plt.bar(categories, gender_counts, color= ['navy','pink'])
total_count = sum(gender_counts)
for category, count in zip(categories, gender_counts):
    percentage = (count / total_count) * 100
    plt.text(category, count, f'{percentage:.1f}%', ha='center', va='bottom', size= 12)
plt.xlabel('Gender',fontdict={'weight': 'bold'})
plt.ylabel('Frequency',fontdict={'weight': 'bold'})
plt.title('GENDER DISTRIBUTION',fontdict={'weight': 'bold'})
plt.show()

#Occupation Distribution
data.Occupation= pd.Series(data.Occupation)
occupation_counts = data.Occupation.value_counts()
occupations= data.Occupation.unique()

plt.style.use("ggplot")
plt.bar(occupations, occupation_counts, color= ['royalblue','darkviolet','crimson','silver','gold','seagreen','sandybrown','magenta','lavender','indigo','darkorange'])
plt.xticks(rotation=45)
for i, v in enumerate(occupation_counts):
    plt.text(i, v, str(v), ha='center', va='bottom', size= 12)
plt.xlabel('Occupations',fontdict={'weight': 'bold'})
plt.ylabel('Frequency',fontdict={'weight': 'bold'})
plt.title('OCCUPATION DISTRIBUTION',fontdict={'weight': 'bold'})
plt.show()


#### MULTIPLE LINEAR REGRESSION MODEL
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats

response = data['Quality of Sleep']
predictors = data[['Physical Activity Level','Sleep Duration', 'Stress Level','Heart Rate','Age']]

model = LinearRegression()
model.fit(predictors, response)

# The Coefficients and Intercept
print("Coefficients:")
for i, coef in enumerate(model.coef_, 1):
    print(f"- B{i}: {coef:.2f}")
print()
print("Intercept:", f"{model.intercept_:.2f}")
print()

# Predictions
y_pred = model.predict(predictors)

# R-squared
r_squared = r2_score(response, y_pred)
print("R-squared:", f"{r_squared:.2f}")
print()

n = len(response)  # number of samples
p = predictors.shape[1]  # number of features

# RSS
rss = np.sum((response - y_pred) ** 2)

# Root MSE
rmse = np.sqrt(rss / (n - p - 1))

# Standard errors of coefficients
se = rmse * np.sqrt(np.diag(np.linalg.inv(np.dot(predictors.T, predictors))))
print("Standard errors of coefficients:")
for se in se:
    print(f"- {se:.3f}")
print()

# t-statistic and p-value
t_statistic = model.coef_ / se
p_values = 2 * (1 - stats.t.cdf(np.abs(t_statistic), n - p - 1))
print("P-values:")
for p in p_values:
    print(f"- {p:.10f}")
print()

# Significant coefficients
significance_level = 0.05
significant_coefficients = predictors.columns[p_values < significance_level]

print("Significant coefficients at 0.05 significance level:")
for coefficient in significant_coefficients:
    print(f"- {coefficient}")


#### SUPPORT VECTOR CLASSIFICATION MODEL
from sklearn.model_selection import train_test_split
data['BMI Category'] = data['BMI Category'].str.replace('Normal Weight', 'Normal')
data.Gender=[ 0 if each == "Male" else 1 for each in data.Gender]

y= data['BMI Category']
x= data[['Gender','Age','Sleep Duration','Quality of Sleep','Physical Activity Level','Stress Level','Heart Rate','Daily Steps']]
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size= 0.2, random_state= 50)

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns

svc= SVC(kernel = 'linear').fit(x_train, y_train)
svc.predict(x_train)

y_prediction= svc.predict(x_test)

cm = confusion_matrix(y_test, y_prediction)

cm_df = pd.DataFrame(cm,
                     index = ['Normal', 'Obese', 'Overweight'], 
                     columns = ['Normal', 'Obese', 'Overweight'])

sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

print("Accuracy:",accuracy_score(y_test, y_prediction))
print("Precision:",precision_score(y_test, y_prediction, average= 'micro'))
print("Recall:",recall_score(y_test, y_prediction, average= 'micro'))