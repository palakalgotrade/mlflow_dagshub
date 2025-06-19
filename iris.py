import mlflow.artifacts
import mlflow.sklearn
import pandas as pd
from  sklearn.datasets import load_iris
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score , confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# print(mlflow.get_tracking_uri())
import dagshub
dagshub.init(repo_owner='palakalgotrade', repo_name='mlflow_dagshub', mlflow=True)

mlflow.set_tracking_uri( 'https://dagshub.com/palakalgotrade/mlflow_dagshub.mlflow' )


iris = load_iris()
X = iris.data
y = iris.target


X_train , X_test , y_trian , y_test = train_test_split(X,y, random_state= 42 , test_size=0.2)

max_depth = 3


mlflow.set_experiment('train_dt')

with mlflow.start_run():
    dt = DecisionTreeClassifier( max_depth=max_depth )
    dt.fit(X_train,y_trian)
    y_pred = dt.predict(X_test)

    accuracy = accuracy_score(y_test , y_pred)

    # logging pararmeter
    mlflow.log_metric('accuracy' ,accuracy) 

    # logging metrics
    mlflow.log_param( 'max_depth' ,max_depth)
    print('accuracy', accuracy)

    #creating a graph
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True , fmt='d' , cmap= 'Blues' , xticklabels= iris.target_names , yticklabels= iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Pridicted')
    plt.title('MATRIX')
    plt.savefig('confusion.png')
    
    ## loging artifact
    mlflow.log_artifact('confusion.png')
    
    ## log code
    mlflow.log_artifact(__file__)
    signature = infer_signature(X_train , dt.predict(X_train))
    
    ## log model
    mlflow.sklearn.log_model(dt,"decision tree",signature=signature)

    ## generic model # but the specific sklearn model gets some more data about meta data
    #mlflow.load_model(dt,"decision_tree")

    # adding tags

    mlflow.set_tag('author','palak')
    mlflow.set_tag('experiment','IRIS_dataset')
    mlflow.set_tag('model','decision tree')
    