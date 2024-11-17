# Personal Finance Predictor
Personal Finance Predictor is a machine learning model designed to analyze your current spending habits and provide the following insights:

1. Categorize users into one of three spending habit categories:
   - Family Focused
   - High Savers
   - Balanced Savers
2. Predict your monthly savings by comparing your age, city tier, and other factors with similar users.
3. Forecast savings across different spending categories.

## Methodology

- Clustering: Utilized the K-Means algorithm to group users into spending habit categories.

- Prediction: Leveraged the Random Forest Regressor to estimate monthly and categorical savings.

- Deployment: Saved the trained models and integrated them into a Streamlit app for user interaction.

## Output
Here is an example output from the application:
![Screenshot (2260)](https://github.com/user-attachments/assets/96022032-4e4f-4069-b6b1-f8126f25960f)

![Screenshot (2261)](https://github.com/user-attachments/assets/f888ef8c-87db-4da7-a35c-0d1a2b5b0df9)
