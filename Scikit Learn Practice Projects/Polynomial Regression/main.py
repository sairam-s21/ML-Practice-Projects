import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def polynomial_regresssion(csv_file, feature_column, label_column, deg, image, predict_feature):
    
    #reading the dataset
    df=pd.read_csv(csv_file)

    #seperating features and label
    X=df.drop(columns=[label_column])
    y=df[label_column]

    #applying polynomial transform of input features
    poly=PolynomialFeatures(degree=deg)
    X_poly=poly.fit_transform(X)

    #creating linear regression model
    model=LinearRegression()
    model.fit(X_poly,y)

    #predicting output for a testcase
    output=model.predict(poly.transform([[predict_feature]]))[0]
    print(f"Input Features : {predict_feature}, predicted output: {output}") 
    
    #calculating mean square error of the model
    y_pred=model.predict(X_poly)
    mse=mean_squared_error(y,y_pred)
    print(f"Mean Square Error of the model: {mse}")

    #plotting the curve and saving as png
    plt.scatter(X,y,color='red',label='Actual Data Points')
    X_grid=np.linspace(min(X.values),max(X.values),100)
    plt.plot(X_grid,model.predict(poly.transform(X_grid)),color="blue",label="Polynomial Regression Curve")
    plt.title(f"Polynomial Regression: {feature_column} vs {label_column}")
    plt.xlabel(feature_column)
    plt.ylabel(label_column)
    plt.legend()
    plt.savefig(image)
    plt.close()

#main function
def main():
    polynomial_regresssion("students.csv","Hours_Studied","Exam_Score",2,"polynomial_regression_curve.png",7)

if __name__=="__main__":
    main()