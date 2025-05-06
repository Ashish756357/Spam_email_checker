import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split

# Load model and vectorizer
model = joblib.load(r'E:\ML project\spam_classifier_model.pkl')
vectorizer = joblib.load(r'E:\ML project\tfidf_vectorizer.pkl')

# Load dataset
df = pd.read_csv('spam_ham_dataset.csv')
if 'label' not in df.columns or 'text' not in df.columns:
    df.columns = ['label', 'text'] + list(df.columns[2:])
df = df[['label', 'text']].dropna()
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Split again for visualization
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label_num'], test_size=0.25, random_state=42)
X_test_vec = vectorizer.transform(X_test)

# ---------- 1. Confusion Matrix ----------
y_pred = model.predict(X_test_vec)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam'])

plt.figure(figsize=(6, 4))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()

# ---------- 2. ROC Curve ----------
y_proba = model.predict_proba(X_test_vec)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="darkorange")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

# ---------- 3. Top Predictive Words ----------
# Get feature importance (highest weights)
feature_names = vectorizer.get_feature_names_out()
class_log_prob = model.feature_log_prob_[1]  # class 1 = spam
top_indices = class_log_prob.argsort()[-20:][::-1]
top_words = feature_names[top_indices]
top_weights = class_log_prob[top_indices]

plt.figure(figsize=(10, 5))
sns.barplot(x=top_weights, y=top_words, palette="magma")
plt.title("Top 20 Predictive Words for Spam")
plt.xlabel("Log Probability")
plt.ylabel("Words")
plt.tight_layout()
plt.show()
