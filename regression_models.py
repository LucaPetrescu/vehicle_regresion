from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": GridSearchCV(Ridge(), param_grid={'alpha': [0.1, 1.0, 10.0]}, cv=5),
    "Lasso Regression": GridSearchCV(Lasso(), param_grid={'alpha': [0.1, 1.0, 10.0]}, cv=5),
    "Decision Tree": GridSearchCV(DecisionTreeRegressor(), param_grid={'max_depth': [3, 5, 10]}, cv=5),
    "Random Forest": GridSearchCV(RandomForestRegressor(), param_grid={'n_estimators': [50, 100, 200]}, cv=5),
    "Extra Trees": GridSearchCV(ExtraTreesRegressor(), param_grid={'n_estimators': [50, 100, 200]}, cv=5),
    "AdaBoost": GridSearchCV(AdaBoostRegressor(), param_grid={'n_estimators': [50, 100, 200]}, cv=5),
    "Gradient Boosting": GridSearchCV(GradientBoostingRegressor(), param_grid={'n_estimators': [50, 100, 200]}, cv=5)
}