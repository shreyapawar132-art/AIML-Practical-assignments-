# Assignment 10: Fake News Detection from Multimodal Data
# Text + image features are fused and classified with Logistic Regression.
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DATA_DIR = Path("fake_news_images")
DATA_DIR.mkdir(exist_ok=True)

# Create small paired demo images so the practical runs without external files.
samples = [
    ("Government announces a new public transport timetable.", "real"),
    ("University releases examination schedule.", "real"),
    ("City opens a new public library for students.", "real"),
    ("School announces revised admission dates.", "real"),
    ("Local hospital publishes new visiting hours.", "real"),
    ("Public department releases official service notice.", "real"),
    ("Miracle drink guarantees instant recovery from every disease.", "fake"),
    ("Viral post says one setting makes internet free.", "fake"),
    ("Secret trick promises unlimited money overnight.", "fake"),
    ("Magic tablet claims to cure every illness instantly.", "fake"),
    ("Forward says everyone will receive free cash today.", "fake"),
    ("Amazing product promises impossible results in one day.", "fake"),
]

text_data = [x[0] for x in samples]
labels = np.array([1 if x[1] == "real" else 0 for x in samples])

image_paths = []
for i, (_, label) in enumerate(samples):
    path = DATA_DIR / f"sample_{i:02d}_{label}.png"
    if not path.exists():
        img = Image.new("RGB", (160, 100), (210, 210, 210) if label == "real" else (70, 70, 70))
        draw = ImageDraw.Draw(img)
        draw.rectangle((10, 10, 150, 90), outline=(255, 255, 255), width=3)
        draw.text((45, 43), label.upper(), fill=(255, 255, 255))
        img.save(path)
    image_paths.append(path)

tfidf = TfidfVectorizer(max_features=40, ngram_range=(1, 2))
T = tfidf.fit_transform(text_data).toarray()

def image_features(path):
    img = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    mean = img.mean(axis=(0, 1))
    std = img.std(axis=(0, 1))
    gray = img.mean(axis=2).mean()
    return np.r_[mean, std, gray]

V = np.array([image_features(p) for p in image_paths])
X = np.hstack([T, V])

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.30, random_state=42, stratify=labels
)

model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
pred = model.predict(X_test)

print("Samples:", len(samples))
print("Combined Features:", X.shape[1])
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Precision:", round(precision_score(y_test, pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, pred, zero_division=0), 4))
print("F1:", round(f1_score(y_test, pred, zero_division=0), 4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
