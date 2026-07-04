from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier


def train_model(X_train, y_train):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7
    )

    X_train_vector = vectorizer.fit_transform(X_train)

    model = PassiveAggressiveClassifier(max_iter=1000)

    model.fit(X_train_vector, y_train)

    return model, vectorizer


def predict_news(model, vectorizer, news):

    news_vector = vectorizer.transform([news])

    prediction = model.predict(news_vector)[0]

    confidence = model.decision_function(news_vector)[0]

    confidence = abs(confidence)

    return prediction, confidence