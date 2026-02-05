# ATP Tennis Match Outcome Prediction

This project predicts the winner of ATP tennis matches using historical match data from 2000 to 2023. The dataset is sourced from [Kaggle: ATP Tennis 2000-2023 (Daily Pull)](https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull). The prediction task is framed as a binary classification problem: predicting whether **Player 1** will win a match.

---

## Dataset

The dataset includes match-level statistics such as:

- Tournament, Series, Court, Surface, Round
- Player rankings and points
- Betting odds
- Match results

For this project, the CSV file `atp_tennis.csv` is used. All rows with missing or invalid data are removed during preprocessing.

---

## Features

The following features are used in the model:

- **Categorical**: `Series`, `Court`, `Surface`, `Round`, `Player_1`, `Player_2`
- **Numerical**: `Best of`, `Rank_diff`, `Pts_diff`, `Odd_1`, `Odd_2`  

Where:
- `Rank_diff` = Player_1's ranking − Player_2's ranking  
- `Pts_diff` = Player_1's points − Player_2's points  

---

## Models

Currently, a **Logistic Regression** classifier is used in a pipeline with preprocessing:

1. **StandardScaler** for numerical features  
2. **OneHotEncoder** for categorical features  

The model is evaluated using **stratified 5-fold cross-validation** with **macro F1-score** as the main metric.

---

