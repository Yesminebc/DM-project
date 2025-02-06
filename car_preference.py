from sklearn.model_selection import train_test_split
from choice_learn.datasets import load_car_preferences
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

### data
data = load_car_preferences(as_frame=True)

# data contains the following columns:
choice_column = 'choice'
types = ['type1', 'type2', 'type3', 'type4', 'type5', 'type6']
fuels = ['fuel1', 'fuel2', 'fuel3', 'fuel4', 'fuel5', 'fuel6']

customer_features = ['college', 'hsg2', 'coml5']

# initialize a new data frame to store the flattened data
flattened_data = pd.DataFrame()

for i in range(6):
    temp = pd.DataFrame()


    temp['rownames'] = data['rownames']
    temp['type'] = data[types[i]]
    temp['fuel'] = data[fuels[i]]
    temp['price'] = data[f'price{i+1}']
    temp['range'] = data[f'range{i+1}']
    temp['speed'] = data[f'speed{i+1}']
    temp['acc'] = data[f'acc{i+1}']
    temp['pollution'] = data[f'pollution{i+1}']
    temp['size'] = data[f'size{i+1}']
    temp['space'] = data[f'space{i+1}']
    temp['cost'] = data[f'cost{i+1}']
    temp['station'] = data[f'station{i+1}']

    for feature in customer_features:
        temp[feature] = data[feature]
    # signify whether the one chosen
    temp['chosen'] = (data[choice_column] == f'choice{i+1}').astype(int)

    flattened_data = pd.concat([flattened_data, temp], axis=0)

flattened_data.reset_index(drop=True, inplace=True)


print(flattened_data.head())


flattened_data['price_per_range'] = flattened_data['price'] / (flattened_data['range'] + 1e-5)  # 价格/续航
flattened_data['acc_per_speed'] = flattened_data['acc'] / (flattened_data['speed'] + 1e-5)  # 加速度/速度
flattened_data['space_per_size'] = flattened_data['space'] / (flattened_data['size'] + 1e-5)  # 空间/尺寸
flattened_data['pollution_per_cost'] = flattened_data['pollution'] / (flattened_data['cost'] + 1e-5)  # 污染/维护成本
flattened_data['price_per_speed'] = flattened_data['price'] / (flattened_data['speed'] + 1e-5)  # 价格/速度
flattened_data['cost_per_range'] = flattened_data['cost'] / (flattened_data['range'] + 1e-5)  # 维护成本/续航
flattened_data['price_per_space'] = flattened_data['price'] / (flattened_data['space'] + 1e-5)  # 价格/空间
flattened_data['station_per_range'] = flattened_data['station'] / (flattened_data['range'] + 1e-5)  # 充电站比例/续航


flattened_data = pd.get_dummies(flattened_data, columns=['type', 'fuel'], drop_first=True)

X = flattened_data[customer_features + [
    'price', 'range', 'speed', 'acc', 'pollution', 'size', 'space', 'cost', 'station',
    'price_per_range', 'acc_per_speed', 'space_per_size', 'pollution_per_cost',
    'price_per_speed', 'cost_per_range', 'price_per_space', 'station_per_range'  # 添加交互特征
] + [col for col in flattened_data.columns if col.startswith('type_') or col.startswith('fuel_')]]
y = flattened_data['chosen']


print("X shape:", X.shape)
print("y shape:", y.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split the train set and test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

##### data #####

##### model #####
# train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# predict
y_pred = model.predict(X_test)
# evaluation
print("Logistic Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# analysis for feature importance
importance = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
print("Feature Importance:\n", importance)
##### model #####

##### feature selection #####
# evaluation the feature importance by random forest
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# gain the importance of feature
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# print
print("Feature Importance in randomforest:\n", feature_importance)

# visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar')
plt.title("Feature Importance - Random Forest")
plt.show()

# set a treshold for features selection
threshold = 0.01
important_features = feature_importance[feature_importance > threshold].index

print(f"Selected Features ({len(important_features)}):", important_features)

# keep the important features
X_train_selected = pd.DataFrame(X_train, columns=X.columns)[important_features]
X_test_selected = pd.DataFrame(X_test, columns=X.columns)[important_features]
##### feature selection #####

#### resample #####
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
print(pd.Series(y_train_resampled).value_counts())  # 确保类别均衡
##### resample #####

##### model after feature selection #####
# train by logistic regression
model_selected = LogisticRegression(max_iter=1000)
# model_selected.fit(X_train_selected, y_train)
model_selected.fit(X_train_resampled, y_train_resampled)
###### model after feature selection #####







