# %% 
#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# 1. Dataset
wine = pd.read_csv("wine.csv", index_col=0)
# %%
wine.head()
print(wine.columns)
# %%
#Clean
wine.columns = wine.columns.str.strip().str.lower()
#Keep wanted columns
wine = wine[["country", "province", "taster_name", "points", "price"]]
#Convert to numeric
wine["points"] = pd.to_numeric(wine["points"])
wine["price"]  = pd.to_numeric(wine["price"])
#Drop missing value rows
wine = wine.dropna()
wine.head
# %%
#2. Exploratory data analysis
#Price
print(wine["price"].describe())
# %%
#KDE on price
wine["price"].plot(kind="kde")
plt.title("Price Distribution")
plt.xlabel("Price ($)")
plt.show()
# %%
#Country
print(wine.groupby("country")["price"].describe())
# %%
#Price by country
avg_price = wine.groupby("country")["price"].mean().sort_values(ascending=False)
print(avg_price.head(10))
#Features picked: Price and points
# %%
#3. Split
#Features
X = wine.drop(columns=["price"])
y = wine["price"]
#Numeric
X_num = X[["points"]]      
X_all = pd.get_dummies(X, drop_first=True)  
#Split
X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_num, y, test_size=0.2, random_state=40
)
 
X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=40
)
# %%
#4. Regressions
#Outcomes
results = {}
#Regression with points only
model1 = LinearRegression()
model1.fit(X_train_num, y_train)
pred1 = model1.predict(X_test_num)
 
results["Numeric (points only)"] = {
    "r2":   r2_score(y_test, pred1),
    "rmse": np.sqrt(mean_squared_error(y_test, pred1))
}
print(results)
#Results
print("R2: 0.1761")
print("RMSE: $37.67")
# %%
#Regression with categorical only (province, country, taste)
X_cat        = pd.get_dummies(X.select_dtypes(exclude=np.number), drop_first=True)
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
    X_cat, y, test_size=0.2, random_state=40
)
 
model2 = LinearRegression()
model2.fit(X_train_cat, y_train_cat)
pred2 = model2.predict(X_test_cat)
 
results["Categorical only"] = {
    "r2":   r2_score(y_test_cat, pred2),
    "rmse": np.sqrt(mean_squared_error(y_test_cat, pred2))
}
print(results)
#Results
print("R2: 0.1001")
print("RMSE: $39.37")

# %%
#Regression with combined model
model3 = LinearRegression()
model3.fit(X_train_all, y_train)
pred3 = model3.predict(X_test_all)
 
results["Combined (all features)"] = {
    "r2":   r2_score(y_test, pred3),
    "rmse": np.sqrt(mean_squared_error(y_test, pred3))
}

print(results)
#Results
print("R2: 0.2370")
print("RMSE: $36.25")

# %%
from sklearn.preprocessing import PolynomialFeatures
# %%
#Polynomial model regression
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_num)   # Transforms [x] into [1, x, x^2, x^3]
 
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
    X_poly, y, test_size=0.2, random_state=40
)
 
model4 = LinearRegression()
model4.fit(X_train_poly, y_train_poly)
pred4 = model4.predict(X_test_poly)
 
results["Polynomial deg-3 (points)"] = {
    "r2":   r2_score(y_test_poly, pred4),
    "rmse": np.sqrt(mean_squared_error(y_test_poly, pred4))
}
 
print(results)
print("R2: 0.2817")
print("RMSE: $35.17")
# %%
#5. 
print("The polynomial model performed the best. It had the highest R squared and the lowest RMSE. This is most likely because it combined all the features and was able to show the nonlinear relationship between the price and critic scores (convex shape).")
# %%
# 6. 
print("I learned that there are different best models depending on the dataset and what kind of relationship is in the variables. I learned that changing the degrees for polynomial models can help or hurt the model depending on what it is, and how to find the best degree for my models. I also learned that combining features can help improve a model's outcomes. From this data, I learned that the points (critics scores) affect the price the most, but the price shoots up when the critic score gets high and it is nonlinear. This is where a polynomial model can be better. I learned that the categorical variables did not affect the predictions that much, although it still improved with the addition of them.")
# %%
