# kaggle-titanic
Kaggle titanic learning competition

This is onboarding project to start participating in Kaggle competitions. The goal is to learn best pratices and how to
develop understanding of participaiton rules.

## Process

Start by understanding the data and the problem statement.

### Exploratory Data Analysis (EDA)

The data is located in `data/` directory. And the inital code is located in `src/eda.py` file.


```shell
# Download dataset
kaggle competitions download -c titanic
kaggle competitions submit -c titanic -f submission.csv -m "Message"
```

# Submissions

This is table of submissions and their results.

| Submission | Description | Result |
|------------|-------------|--------|
| 1          | Initial submission |  |


 The submission format is:
 ```
 PassengerId,Survived
892,0
893,1
894,0
 ```
 Where the `PassengerId` is the id of the passenger and `Survived` is the predicted survival status.
The PassengerId must same from `test.csv` file, in same order as in `test.csv` file.
