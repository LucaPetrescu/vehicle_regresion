from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from regression_models import models
import matplotlib.pyplot as plt
import numpy as np

class LinearRegression:

    def __init__(self, df, features, target, n_folds):
        self.df = df
        self.features = features
        self.target = target
        self.n_folds = n_folds

    def split_data(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.features, self.target, test_size=0.25)
        X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
        return X_train, X_test, Y_train, Y_test, self.n_folds
    
    def cross_rmse_train(self, model):
        X_train, X_test, Y_train, Y_test, n_folds = self.split_data()
        kf = KFold(n_folds, shuffle=True).get_n_splits(self.df.values)
        rmse = np.sqrt(-1 * cross_val_score(model, X_train, Y_train, scoring="neg_mean_squared_error", cv=kf))
        return rmse
    
    def cross_rmse_test(self, model):
        X_train, X_test, Y_train, Y_test, n_folds = self.split_data()
        kf = KFold(n_folds, shuffle=True).get_n_splits(self.df.values)
        rmse = np.sqrt(-1 * cross_val_score(model, X_test, Y_test, scoring="neg_mean_squared_error", cv=kf))
        return rmse


    def run_regression(self, model_name):
        if model_name in models:
            X_train, X_test, Y_train, Y_test, n_folds = self.split_data()
            model = models[model_name]
            model.fit(X_train, Y_train)

            scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='neg_mean_squared_error')

            mse_scores = -scores
            mean_mse = mse_scores.mean()
            std_mse = mse_scores.std()

            best_params = model.best_params_ if hasattr(model, 'best_params_') else {}

            model.fit(X_train, Y_train)
            y_pred = model.predict(X_train)
            r_squared = r2_score(Y_train, y_pred)

            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)

            train_rmse = self.cross_rmse_train(model).mean()
            test_rmse = self.cross_rmse_test(model).mean()

            results = {
                "model_name": model_name,
                "best_params": best_params,
                "mean_mse": mean_mse,
                "std_mse": std_mse,
                "r_squared": r_squared,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse
            }

            plt.scatter(train_predictions, Y_train, c = "blue",  label = "Training data")
            plt.scatter(test_predictions, Y_test, c = "black",  label = "Validation data")
            plt.title("Linear regression")
            plt.xlabel("Predicted values")
            plt.ylabel("Real values")
            plt.legend(loc = "upper left")
            plt.plot([train_predictions.min(), train_predictions.max()], [train_predictions.min(), train_predictions.max()], c = "red")
            plt.show()

            print(f"Model: {model_name}")
            print(f"Best Params: {best_params}")
            print(f"MSE: {mean_mse}")
            print(f"R-squared: {r_squared}\n")
            print(f"Train RMSE: {train_rmse}\n")
            print(f"Test RMSE: {test_rmse}\n")

            return results
        else:
            print(f"Model {model_name} not found.")
            return None


