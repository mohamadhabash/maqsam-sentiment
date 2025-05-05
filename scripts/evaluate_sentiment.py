import csv
from sklearn.metrics import accuracy_score, f1_score, classification_report
from app.predictor import SentimentPredictor

def load_test_set(path):
    summaries, labels = [], []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            summaries.append(row['summary'])
            labels.append(row['label'])
    return summaries, labels

def evaluate(path):
    summaries, true_labels = load_test_set(path)
    predictor = SentimentPredictor()
    pred_labels = [predictor.predict(text) for text in summaries]
    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    print(f"Results for {path}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(true_labels, pred_labels, target_names=['Positive','Neutral','Negative']))


if __name__ == "__main__":
    evaluate("data/test_set_en.csv")
    evaluate("data/test_set_ar.csv")
