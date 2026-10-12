# Assignment 4: Naive Bayes Classification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Labeled sample messages used in the practical
messages = [
    "Hey, are we still meeting at 5 today?",
    "Please call me when you reach home.",
    "Can you send the notes for today's lecture?",
    "Your appointment is confirmed for tomorrow.",
    "Don't forget to bring your project file.",
    "You have been chosen for a cash reward. Claim now.",
    "Free prize waiting for you. Text CLAIM immediately.",
    "Earn money fast from home. Join our special offer now.",
    "The meeting is scheduled for Monday.",
    "Please submit the assignment before Friday.",
    "Congratulations! You won a free vacation prize.",
    "Get cash bonus now by clicking this link."
]
labels = [
    "ham", "ham", "ham", "ham", "ham", "spam",
    "spam", "spam", "ham", "ham", "spam", "spam"
]

X_train, X_test, y_train, y_test = train_test_split(
    messages, labels, test_size=0.25, random_state=42, stratify=labels
)

vectorizer = CountVectorizer()
A = vectorizer.fit_transform(X_train)
B = vectorizer.transform(X_test)

model = MultinomialNB().fit(A, y_train)
pred = model.predict(B)

print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Precision:", round(precision_score(y_test, pred, pos_label="spam", zero_division=0), 4))
print("Recall:", round(recall_score(y_test, pred, pos_label="spam", zero_division=0), 4))
print("F1:", round(f1_score(y_test, pred, pos_label="spam", zero_division=0), 4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred, labels=["ham", "spam"]))

cm = confusion_matrix(y_test, pred, labels=["ham", "spam"])
plt.imshow(cm, cmap="Blues")
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["Ham", "Spam"])
plt.yticks([0, 1], ["Ham", "Spam"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout()
plt.show()
