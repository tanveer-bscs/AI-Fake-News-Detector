import pandas as pd
from sklearn.model_selection import train_test_split
from modules.detector import train_model, predict_news
from sklearn.metrics import accuracy_score
from modules.cleaner import clean_text
from modules.history import save_history
from modules.report_generator import generate_report

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

print("=" * 50)
print("AI Fake News Detector")
print("=" * 50)

print("\nFake News Dataset")
print(fake.head())

print("\nTrue News Dataset")
print(true.head())

print("\nFake Dataset Shape :", fake.shape)
print("True Dataset Shape :", true.shape)

print("\nFake Dataset Columns")
print(fake.columns)

print("\nTrue Dataset Columns")
print(true.columns)

# Add Labels
fake["label"] = 0
true["label"] = 1

# Merge Datasets
news = pd.concat([fake, true], ignore_index=True)

print("\nTotal Dataset Shape :", news.shape)

print("\nDataset Labels")
print(news["label"].value_counts())

print("\nFirst 5 Rows")
print(news.head())

# Shuffle Dataset
news = news.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nDataset Shuffled Successfully!")

print("\nFirst 5 Rows After Shuffle")
print(news.head())

# Remove Unnecessary Columns
news = news.drop(["subject", "date"], axis=1)

print("\nColumns After Removing Unnecessary Data")
print(news.columns)

print("\nUpdated Dataset")
print(news.head())

# Combine Title and Text

news["content"] = news["title"] + " " + news["text"]

print("\nCombined Dataset")

print(news[["content", "label"]].head())

# Clean News Content

news["content"] = news["content"].apply(clean_text)

print("\nCleaned Dataset")
# Features and Labels

X = news["content"]
y = news["label"]

print("\nFeatures and Labels Ready!")
print("Total Samples :", len(X))

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

print(news[["content", "label"]].head())

# Train Model

model, vectorizer = train_model(X_train, y_train)

X_train_vector = vectorizer.transform(X_train)
X_test_vector = vectorizer.transform(X_test)

print("\nModel Trained Successfully!")
print("TF-IDF Vectorization Completed!")
print("Training Data Shape :", X_train_vector.shape)
print("Testing Data Shape  :", X_test_vector.shape)

# Predict

prediction = model.predict(X_test_vector)

print("\nPrediction Completed!")

# Accuracy

accuracy = accuracy_score(y_test, prediction)

print("\nModel Accuracy")

print(f"Accuracy : {accuracy*100:.2f}%")

print("\n" + "=" * 50)
print("Check Your Own News")
print("=" * 50)

user_news = input("\nPaste News:\n\n")

user_news = clean_text(user_news)

prediction, confidence = predict_news(
    model,
    vectorizer,
    user_news
)

print("\nPrediction")
print("-" * 30)

if prediction == 1:
    print("Result : REAL NEWS")
else:
    print("Result : FAKE NEWS")

print(f"Confidence Score : {confidence:.2f}")

if confidence >= 2:
    risk = "LOW"

elif confidence >= 1:
    risk = "MEDIUM"

else:
    risk = "HIGH"

print("Risk Level      :", risk)

result = "REAL NEWS" if prediction == 1 else "FAKE NEWS"

print("\nRecommendation")
print("-" * 30)

if prediction == 1:
    print("This news appears to be genuine. Always verify with trusted sources.")
else:
    print("This news may be misleading or fake. Verify it before sharing.")

save_history(
    user_news,
    result,
    confidence,
    risk
)

print("\nHistory saved successfully!")

generate_report(
    user_news,
    result,
    confidence,
    risk
)

print("Report generated successfully!")