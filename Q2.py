# %%
#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
#Load dataset
cars = pd.read_csv("cars_hw.csv")
#cars = cars.dropna()
# %%
cars.head()
print(cars.columns)
# %%
#1 Clean
cars.columns = cars.columns.str.strip().str.lower()
cars = cars.drop(columns=["unnamed: 0"])
cars["no_of_owners"] = cars["no_of_owners"].str.replace(r"[a-zA-Z]", "", regex=True)
cars["no_of_owners"] = cars["no_of_owners"].astype(int)
cars["make_year"] = pd.to_numeric(cars["make_year"])
cars["mileage_run"] = pd.to_numeric(cars["mileage_run"])
cars["seating_capacity"] = pd.to_numeric(cars["seating_capacity"])
cars["price"] = pd.to_numeric(cars["price"])
cars = cars.dropna()

# %%
cars.head()
# %%
#2 Summarize price variable
print("Summary of Price:", cars["price"].describe())
# %%
#Kernel density plot
cars["price"].plot(kind="kde")
plt.title("Price Distribution")
plt.show()
# %%
#Group by make
print("Price by Make:", cars.groupby("make")["price"].describe())
# %%
#KDE for make
for m in cars["make"].unique():
    subset = cars[cars["make"] == m]
    subset["price"].plot(kind="kde", label=m)

plt.title("Price Distribution by Car Brand")
plt.xlabel("Price")
plt.legend()
plt.show()
# %%
#Most expensive
avg_price = cars.groupby("make")["price"].mean().sort_values(ascending=False)

print("Average price by brand:")
print(avg_price)

print("Most expensive brands have the highest average price which include MG Motors, Kia, and Jeep.")
print("Prices look skewed right and the average prices look very large.")
# %%
#3 Split into train and test
X = cars.drop(columns=["price"])
y = cars["price"]
X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=40)

# %%
#4 Make a model
model = LinearRegression()
#Include only numeric values
X_num = X.select_dtypes(include=np.number)
#Train
model.fit(X_train, y_train)
#Predict
pred = model.predict(X_test)
#R^2 and RMSE
print("Numeric Model R2:", r2_score(y_test, pred))
print("Numeric Model RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
# %%
#Include only categorical
X_cat = pd.get_dummies(X.select_dtypes(exclude=np.number), drop_first=True)
#Split
X_train, X_test, y_train, y_test = train_test_split(X_cat, y, test_size=0.2, random_state=40)
#Model
model = LinearRegression()
#Train
model.fit(X_train, y_train)
#Predict
pred = model.predict(X_test)
#R2 and RMSE
print("Categorical Model R2:", r2_score(y_test, pred))
print("Categorical Model RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
# %%
print('The categorical model performed better.')
# %%
#Combined
X_all = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=40)
#Model
model = LinearRegression()
#Fit
model.fit(X_train, y_train)
#Predict
pred = model.predict(X_test)
#R2 and RMSE
print("Combined Model R2:", r2_score(y_test, pred))
print("Combined Model RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
# %%
print("The conbined model works the best by about a .1213 difference in the R squared coefficient.")
# %%
from sklearn.preprocessing import PolynomialFeatures
# %%
#5 Polynomial
poly = PolynomialFeatures(degree=4)
#Fit and transform
X_poly = poly.fit_transform(X_num)
#split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=40)
#Model
model = LinearRegression()
model.fit(X_train, y_train)
#Predict
pred = model.predict(X_test)
#R2 and RMSE
print("Polynomial Model R2:", r2_score(y_test, pred))
print("Polynomial Model RMSE:", np.sqrt(mean_squared_error(y_test, pred)))

# %%
print('Increasing the degree decreases the R2 and increase the RMSE, making the model worse as the degree increases.')
# %%
print("R squared became negative at 24 degrees. The best model's R2 was 0.33505268178533565 and had an RMSE of 259209.20476707627. This is wors ethan my best model from part 4.")
# %%
#6 best model
X_all = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=40)
#Model
model = LinearRegression()
#Fit
model.fit(X_train, y_train)
#Predict
pred = model.predict(X_test)
pred = model.predict(X_test)
#Plot
plt.scatter(y_test, pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs Actual Prices")

#Diagonal line
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle="--")

plt.show()
# %%
#Residuals
residuals = y_test - pred
#Plot
residuals.plot(kind="kde")
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.show()
# %%
print("Yes, the residuals look roughly bell-shaped around 0, although it could be better.")
print("Strengths of my model is that it shows general price trends and it also does well with the test data. Weaknesses are that it falls to extreme values and it can overlook nonlinear relationships.")
# %%
