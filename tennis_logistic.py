import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

df = pd.read_csv("atp_tennis.csv")

df.replace(-1, pd.NA, inplace=True)
df.replace(-1.0, pd.NA, inplace=True)
df.dropna(inplace=True)

df["target"] = (df["Winner"] == df["Player_1"]).astype(int) # 1 if Player_1 wins, else 0
df = df.drop(columns=["Winner", "Score", "Date", "Tournament"])
df["Rank_diff"] = df["Rank_1"] - df["Rank_2"]
df["Pts_diff"] = df["Pts_1"] - df["Pts_2"]
df = df.drop(columns=["Rank_1", "Rank_2", "Pts_1", "Pts_2"])

categorical_features = ["Series", "Court", "Surface", "Round", "Player_1", "Player_2"]
numeric_features = ["Best of", "Rank_diff", "Pts_diff", "Odd_1", "Odd_2"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),       # scala i numerici
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)  # one-hot encode le categoriche
    ]
)


temp_df = df[numeric_features + ['target']].copy()

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = temp_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

################   SCATTER PLOT   ################
plt.figure(figsize=(8,6))
plt.scatter(df[df["target"]==1]["Rank_diff"], df[df["target"]==1]["Pts_diff"], # select lines where Player_1 wins and display as blue
color='blue', alpha=0.5, label='Player_1 wins')
plt.scatter(df[df["target"]==0]["Rank_diff"], df[df["target"]==0]["Pts_diff"],
            color='red', alpha=0.5, label='Player_2 wins')
plt.xlabel("Rank_1 - Rank_2")
plt.ylabel("Pts_1 - Pts_2")
plt.title("linear separability")
plt.legend()
plt.grid(True)
plt.savefig("plot.png")

#############  END SCATTER PLOT   #################

X = df[categorical_features + numeric_features]
y = df["target"]

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

############### Pipeline with Logistic ################
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    ))
])

############ GridSearchCV for K #############
param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=1,
    verbose=1
)
grid_search.fit(X_trainval, y_trainval)



y_prob = grid_search.best_estimator_.predict_proba(X_test)[:, 1]  # final model probabilities y=1


thresholds = np.linspace(0.3, 0.7, 41) # some thresholds to try
f1_scores = []

for t in thresholds: # iterate over thresholds
    y_pred_t = (y_prob > t).astype(int) # starting from probabilities of y_prob, get predictions with threshold t
    f1 = f1_score(y_test, y_pred_t, average="macro") # compute F1 with y_pred_t
    f1_scores.append(f1) # store F1


best_t = thresholds[np.argmax(f1_scores)]
best_f1 = max(f1_scores) # best F1

print("Best threshold:", best_t)
print("Best F1:", best_f1)


y_pred_best = (y_prob > best_t).astype(int)
print(classification_report(y_test, y_pred_best, digits=4))

