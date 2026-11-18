from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

data = pd.read_csv('train.csv')
x = data.iloc[:, 1:-1]

y = data['Will_Buy_EV'].replace({'Yes': 1, 'No': 0}).astype("int64")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned', 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work']),
        ('cat', OneHotEncoder(), ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Environmental_Concern_Level'])
    ]
)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SGDClassifier(loss="log_loss", max_iter=2000, random_state=17))
])

pipeline.fit(x, y)

# Evaluate
print('Evaluating...')
x_test = pd.read_csv('test.csv')
x_test = x_test.iloc[:, 1:]

prediction = pipeline.predict_proba(x_test)
df = pd.DataFrame(prediction, columns=['A', 'Will_Buy_EV'])
df.insert(0, 'id', range(668665, 668665 + len(df)))
df.to_csv("submission.csv", index=False, columns=['id', 'Will_Buy_EV'], float_format='%.17f')

