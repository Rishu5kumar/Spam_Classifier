# Spam Classifier

## Steps to Build a Spam Classifier

### 1. **Import Libraries**

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
```

### 2. **Load the Dataset**

Load your dataset containing messages and their labels (spam or not spam).

```python
# Sample dataset
data = {
    'message': ['Free money now!!!', 'Hi, how are you?', 'Win a lottery!', 'See you tomorrow.'],
    'label': ['spam', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)
```

### 3. **Preprocess the Data**

Convert labels to binary format and split the dataset.

```python
# Encode labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
```

### 4. **Vectorize the Text**

Transform text data into numerical format using Count Vectorization.

```python
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
```

### 5. **Train the Model**

Use a Naive Bayes classifier to train the model.

```python
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)
```

### 6. **Evaluate the Model**

Make predictions and evaluate model performance.

```python
y_pred = model.predict(X_test_vectorized)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

--- 
