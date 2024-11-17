import streamlit as st
import pandas as pd
import joblib

# Load the saved models
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca_model.pkl')
rf_savings = joblib.load('rf_savings_model.pkl')

# Load the saved expense models dynamically
expense_models = {}
savings_categories = [
    'Potential_Savings_Groceries', 'Potential_Savings_Transport', 'Potential_Savings_Eating_Out', 
    'Potential_Savings_Entertainment'
]
for category in savings_categories:
    expense_models[category] = joblib.load(f'{category}_model.pkl')

# Function to predict and print results
def predict_and_print_results(new_user):
    # Features for savings prediction
    features_savings = [
        'Income', 'Age', 'Dependents', 'Rent', 'Loan_Repayment', 'Groceries', 'Transport', 
        'Eating_Out', 'Entertainment', 'City Tier', 'Occupation_Retired', 
        'Occupation_Self_Employed', 'Occupation_Student'
    ]
    
    # Predict savings
    predicted_savings = rf_savings.predict(new_user[features_savings])[0]

    # Predict potential expenses in various categories
    expense_predictions = {}
    for category, rf_expense in expense_models.items():
        expense_predictions[category] = rf_expense.predict(new_user[features_savings])[0]

    # Calculate ratios for KMeans prediction
    entertainment_ratio = (new_user['Eating_Out'] + new_user['Entertainment']) / new_user['Income']
    family_ratio = (new_user['Groceries'] + new_user['Transport']) / new_user['Income']
    savings_ratio = new_user['Savings'] / new_user['Income']  

    # Prepare the input for KMeans prediction
    cluster_input = pd.DataFrame({
        'entertainment_ratio': [entertainment_ratio],
        'family_ratio': [family_ratio],
        'savings_ratio': [savings_ratio],
    })

    # Scale the input for KMeans and apply PCA transformation
    scaled_input = scaler.transform(cluster_input)
    pca_input = pca.transform(scaled_input)

    # Predict the cluster using KMeans
    predicted_cluster = kmeans.predict(pca_input)[0]
    cluster_names = {0: "Family Focused", 1: "High Savers", 2: "Balanced Savers"}
    predicted_category = cluster_names.get(predicted_cluster, "Unknown Category")

    return {
        'predicted_savings': predicted_savings,
        'expense_predictions': expense_predictions,
        'user_cluster': predicted_cluster,
        'user_category': predicted_category,
    }
st.title("Financial Prediction Model")
st.write("### Enter your details to get financial predictions.")

# User Input
new_user = pd.DataFrame({
    'Income': [st.number_input('Income (₹)', value=4000, min_value=0)],
    'Age': [st.number_input('Age', value=30, min_value=18)],
    'Dependents': [st.number_input('Dependents', value=0, min_value=0)],
    'Rent': [st.number_input('Rent (₹)', value=750, min_value=0)],
    'Loan_Repayment': [st.number_input('Loan Repayment (₹)', value=1000, min_value=0)],
    'Groceries': [st.number_input('Groceries (₹)', value=300, min_value=0)],
    'Transport': [st.number_input('Transport (₹)', value=100, min_value=0)],
    'Eating_Out': [st.number_input('Eating Out (₹)', value=100, min_value=0)],
    'Entertainment': [st.number_input('Entertainment (₹)', value=100, min_value=0)],
    'Savings': [st.number_input('Savings (₹)', value=750, min_value=0)]
})

# Occupation selection using radio buttons
occupation = st.radio(
    "Select your Occupation",
    ("Retired", "Self Employed", "Student")
)

# City Tier selection using radio buttons
city_tier = st.radio(
    "Select your City Tier",
    (0, 1, 2)
)

# Map the occupation to the respective columns
new_user['Occupation_Retired'] = 1 if occupation == "Retired" else 0
new_user['Occupation_Self_Employed'] = 1 if occupation == "Self Employed" else 0
new_user['Occupation_Student'] = 1 if occupation == "Student" else 0
new_user['City Tier'] = city_tier

# Make predictions when the user clicks the button
if st.button("Get Prediction"):
    results = predict_and_print_results(new_user)
    
    # Display the results
    st.subheader("Predicted Results:")

    # Spending Profile
    st.write(f"**Spending Profile Type:** {results['user_category']}")

    # Savings Prediction
    st.write(f"**Recommended Monthly Savings:** ₹{results['predicted_savings']:.2f}")
    
    # Potential Monthly Savings by Category
    st.write("**Potential Monthly Savings by Category:**")
    for category, amount in results['expense_predictions'].items():
        category_name = category.replace('Potential_Savings_', '').replace('_', ' ')
        st.write(f"{category_name.capitalize()}: ₹{amount:.2f}")