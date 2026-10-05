import json
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from scipy.sparse import hstack
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

data = []
with open(
    r"Dataset/LeetCodeDataset-v0.3.1-train.jsonl","r",
    encoding="utf-8"
) as f:

    for line in f:
        obj = json.loads(line)
        data.append({
            "question": obj["problem_description"],
            "difficulty": obj["difficulty"],
            "tags": " ".join(obj["tags"])
        })

df_train = pd.DataFrame(data)
print("Training Samples:", len(df_train))

data = []
with open(
    r"Dataset/LeetCodeDataset-v0.3.1-test.jsonl",
    encoding="utf-8"
) as f:

    for line in f:

        obj = json.loads(line)
        data.append({
            "question": obj["problem_description"],
            "difficulty": obj["difficulty"],
            "tags": " ".join(obj["tags"])
        })

df_test = pd.DataFrame(data)
print("Testing Samples:", len(df_test))

train_tags = df_train["tags"].str.split()
test_tags = df_test["tags"].str.split()
mlb = MultiLabelBinarizer()

X_train_tags = mlb.fit_transform(train_tags)
X_test_tags = mlb.transform(test_tags)

X_train_text = df_train["question"]
y_train = df_train["difficulty"]
X_test_text = df_test["question"]
y_test = df_test["difficulty"]

vectorizer = TfidfVectorizer(
    max_features=30000,
    stop_words="english",
    ngram_range=(1,6),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

X_train = hstack([
    X_train_tfidf,
    X_train_tags
])

X_test = hstack([
    X_test_tfidf,
    X_test_tags
])

print("Feature Shape:", X_train.shape)

model = LinearSVC(
    C = 3.5,
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:", accuracy)
print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nConfusion Matrix:\n")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)

joblib.dump(
    model,
    "difficulty_model.pkl"
)

joblib.dump(
    vectorizer,
    "vectorizer.pkl"
)

print("\nModel Saved!")