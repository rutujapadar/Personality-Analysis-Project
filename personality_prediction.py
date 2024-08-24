import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import joblib
import matplotlib.pyplot as plt

def train_model():
    data = pd.read_csv("train dataset.csv")

    X = data[['openness', 'neuroticism', 'conscientiousness', 'agreeableness', 'extraversion']]
    y = data['Personality (Class label)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, y_train)

    # Visualize the Decision Tree
    plt.figure(figsize=(20, 15), dpi=600)
    plot_tree(model, filled=True, feature_names=X.columns, class_names=model.classes_)
    plt.tight_layout()
    plt.savefig("decision_tree_visualization.png")

    # Save the trained model
    joblib.dump(model, "personality_model.pkl")

if __name__ == "__main__":
    train_model()
