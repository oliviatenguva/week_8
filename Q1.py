# %%
#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# %%
#Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/DS3001/linearRegression/refs/heads/main/data/Q1_clean.csv")
df.head()
# %%
#Q1
#1. Compute avg prices and scores by Neighborhood
#Fix neighbourhood column
df.columns = df.columns.str.strip()
# %%
#Compute average price and rating by neighborhood
grouped = df.groupby("Neighbourhood")[["Price", "Review Scores Rating"]].mean()
print("Average by Neighborhood:\n", grouped)

# %%
#Most expensive neighborhood
most_expensive = grouped["Price"].idxmax()
print("Most expensive neighborhood:", most_expensive)
#Manhattan
# %%
#Kernel density plots
for n in df["Neighbourhood"].unique():
    subset = df[df["Neighbourhood"] == n]
    subset["Price"].plot(kind="kde", label=n)

plt.title("Price Density by Neighbourhood")
plt.legend()
plt.show()

#Log price
for n in df["Neighbourhood"].unique():
    df[df["Neighbourhood"] == n]["Price"].plot(kind="kde", label=n)
plt.title("Price Density")
plt.legend()
plt.show()

df["log_price"] = np.log(df["Price"])

for n in df["Neighbourhood"].unique():
    df[df["Neighbourhood"] == n]["log_price"].plot(kind="kde", label=n)
plt.title("Log Price Density")
plt.legend()
plt.show()

# %%
#2 Regress price without the intercept
X = pd.get_dummies(df["Neighbourhood"], drop_first=False)
y = df["Price"]

model_no_intercept = LinearRegression(fit_intercept=False)
model_no_intercept.fit(X, y)

coeff_no_intercept = pd.Series(model_no_intercept.coef_, index=X.columns)
print("Coefficients (no intercept):", coeff_no_intercept)
print("The averages from part one are the same as regressing price on neighborhood with no intercept.")
print("The coefficients in regression show the average of the countinous variable of each category.")
# %%
#3 Regression with intercept
X2 = pd.get_dummies(df["Neighbourhood"], drop_first=True)

model_intercept = LinearRegression()
model_intercept.fit(X2, y)

print("Intercept:", model_intercept.intercept_)

coeff_intercept = pd.Series(model_intercept.coef_, index=X2.columns)

print("The intercept is the baseline neighborhood's average price.")
print("The coefficients show how much less or more expensive a neighborhood is compared to the baseline (intercept).")
print("To get the coefficients in part two from these, you subtract the intercept and coefficient.")
# %%
#4 Split the sample into training and test
#Run a regression 
X3 = df[["Review Scores Rating", "Neighbourhood"]]
X3 = pd.get_dummies(X3, drop_first=True)

y = df["Price"]

#Split 80/20

X_train, X_test, y_train, y_test = train_test_split(X3, y, test_size=0.2, random_state=40)

#Fit model
model3 = LinearRegression()
model3.fit(X_train, y_train)

#Predict
preds = model3.predict(X_test)

#Scores
r2 = r2_score(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("R^2:", r2)
print("RMSE:", rmse)
print("Coefficient on Review Scores Rating:", rating_coef)
print("The coefficient Review Scores Rating is how much the price increases when rating increases by one point.")
print("The most expensive property will have the largest dummy coefficient.")
#Most Expensive
#Indices to get original properties
original_test_14 = df.iloc[y_test.index]
#Predict prices
pred_14 = model3.predict(X_test)
#Index of max predicted price
idx_max_14 = np.argmax(pred_14)
#Original property info
most_expensive_14 = original_test_14.iloc[idx_max_14]
#Most expensive
print("Most expensive property type: ", most_expensive_14["Property Type"])
# %%
#5 Run a Regression 
X4 = df[["Review Scores Rating", "Neighbourhood", "Property Type"]]
X4 = pd.get_dummies(X4, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X4, y, test_size=0.2, random_state=40)

#Fit
model4 = LinearRegression()
model4.fit(X_train, y_train)

#Predict
preds4 = model4.predict(X_test)

r2_2 = r2_score(y_test, preds4)
rmse_2 = np.sqrt(mean_squared_error(y_test, preds4))

print("R^2:", r2_2)
print("RMSE:", rmse_2)

rating_coef = pd.Series(model4.coef_, index=X4.columns)["Review Scores Rating"]
print("Coefficient on Review Scores Rating:", rating_coef)
# %%
#Most Expensive
#Use indices
original_test_15 = df.iloc[y_test.index] 
#Predict prices
pred_15 = model4.predict(X_test)
#Index of max predicted price
idx_max_15 = np.argmax(pred_15)
#Original property
most_expensive_15 = original_test_15.iloc[idx_max_15]
#Most expensive
print("Most expensive property type: ", most_expensive_15["Property Type"])
# %%
#6 The coefficient on Review Scores Rating changing from 4 to 5 shows that there are more variables being controlled. 5 adds property type, which changes the coefficient. 