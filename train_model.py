import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


# Load data
columns = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income'
]

data = pd.read_csv(
    "adult.data",
    header=None,
    skipinitialspace=True
)

data.columns = columns


# Handle missing values
data = data.replace('?', np.nan)

categorical_cols = data.select_dtypes(
    include='object'
).columns

for col in categorical_cols:
    data[col] = data[col].fillna(
        data[col].mode()[0]
    )


# Remove duplicates
data = data.drop_duplicates()


# Convert target
data['income'] = data['income'].map({
    '<=50K': 0,
    '>50K': 1
})


# X and y
X = data.drop('income', axis=1)
y = data['income']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Columns
categorical_cols = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country'
]

numeric_cols = [
    'age',
    'fnlwgt',
    'education-num',
    'capital-gain',
    'capital-loss',
    'hours-per-week'
]


# Encoding
preprocessor = ColumnTransformer(
    transformers=[
        (
            'num',
            'passthrough',
            numeric_cols
        ),
        (
            'cat',
            OneHotEncoder(handle_unknown='ignore'),
            categorical_cols
        )
    ]
)

X_train_encoded = preprocessor.fit_transform(X_train)


# Scaling
scaler = StandardScaler(with_mean=False)

X_train_scaled = scaler.fit_transform(
    X_train_encoded
)


# SMOTE
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)


# Random Forest
print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train_smote,
    y_train_smote
)


# Save everything
joblib.dump(
    preprocessor,
    "preprocessor.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

joblib.dump(
    rf,
    "random_forest_model.pkl"
)

print("Training completed!")
print("Files saved successfully!")