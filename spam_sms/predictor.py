import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Loading data
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_table(url, header=None, names=['label', 'message'])
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# Prepping for the model
X = df.message
y = df.label_num
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Vectorizing (Turning text to numbers)
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

# Training
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

def classify(text):
    data = vect.transform([text])
    result = nb.predict(data)
    return "SPAM" if result[0] == 1 else "HAM"

if __name__ == "__main__":
    msg = input("Enter a message to test: ")
    print(f"Classification: {classify(msg)}")