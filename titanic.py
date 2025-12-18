import pandas as pd

# Using the 'r' prefix makes the path easy to read for Python
path = r'D:\titanic-dataset\Titanic-Dataset.csv'

# Load the data
df = pd.read_csv(path)

# Display the data to confirm it's working
print("Successfully loaded the Titanic dataset!")
print(df.head())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# 1. Survival by Gender
print("\n--- Survival by Gender ---")
print(df.groupby('Sex')['Survived'].mean())

# 2. Survival by Passenger Class (1st, 2nd, or 3rd Class)
print("\n--- Survival by Ticket Class ---")
print(df.groupby('Pclass')['Survived'].mean())

'''part step 3'''

import seaborn as sns
import matplotlib.pyplot as plt

# 1. Create a "Heatmap" to see where data is missing
# (This looks very professional in a BCA project report!)
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Missing Data Map')
plt.show()

# 2. Survival vs Pclass (The Class Divide)
plt.figure(figsize=(8,5))
sns.countplot(x='Survived', hue='Pclass', data=df, palette='rainbow')
plt.title('Survival Count by Ticket Class')
plt.show()

'''Part 4 '''

def predict_my_survival(pclass, sex, age):
    # Rule 1: High survival for females in upper classes
    if sex.lower() == 'female' and pclass in [1, 2]:
        return "Likely Survived (High Probability)"
    
    # Rule 2: High mortality for males in 3rd class
    elif sex.lower() == 'male' and pclass == 3:
        return "Unlikely to Survive (High Risk)"
    
    # Rule 3: Priority for children
    elif age < 12:
        return "Likely Survived (Child Priority)"
    
    else:
        return "Result is uncertain - it depended on luck and location!"

# --- Let's test it ---
print("\n--- Titanic Survival Predictor ---")
print(f"Me (Class 1, Male, 19): {predict_my_survival(1, 'male', 19)}")
print(f"Scenario (Class 3, Male, 25): {predict_my_survival(3, 'male', 25)}")

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Prepare the data (ML models only understand numbers, not "male/female")
le = LabelEncoder()
df['Sex_n'] = le.fit_transform(df['Sex']) # female=0, male=1

# 2. Select the "Features" (what we use to predict) and the "Target" (what to predict)
# We need to drop rows with missing Age for this simple model to work
clean_df = df[['Survived', 'Pclass', 'Sex_n', 'Age']].dropna()
inputs = clean_df.drop('Survived', axis='columns')
target = clean_df['Survived']

# 3. Create and "Train" the model
model = DecisionTreeClassifier(max_depth=3)
model.fit(inputs.values, target)

# 4. Use the model to predict!
# Example: Class 3, Male (1), Age 25
prediction = model.predict([[3, 1, 25]])
print(f"\nMachine Learning Prediction for 3rd Class Male: {'Survived' if prediction[0] == 1 else 'Did Not Survive'}")

'''IQ level Checking'''
# Check how accurate the model is (out of 1.0)
score = model.score(inputs.values, target)
print(f"\nModel Accuracy: {score * 100:.2f}%")



def check_my_chance():
    print("\n--- TEST YOUR OWN SURVIVAL ---")
    pclass = int(input("Enter Class (1, 2, or 3): "))
    gender = input("Enter Gender (male/female): ")
    age = int(input("Enter Age: "))
    
    # Convert gender to number
    gender_n = 1 if gender.lower() == 'male' else 0
    
    # 1. Get the Raw Prediction (0 or 1)
    res = model.predict([[pclass, gender_n, age]])
    
    # 2. Get the Probabilities [[Chance of 0, Chance of 1]]
    proba = model.predict_proba([[pclass, gender_n, age]])
    
    # The [0][1] index gives the probability of survival (Class 1)
    survival_percent = proba[0][1] * 100
    
    status = "SURVIVED" if res[0] == 1 else "DID NOT SURVIVE"
    
    print("-" * 30)
    print(f"Result: The AI predicts you would have {status}")
    print(f"Confidence/Survival Probability: {survival_percent:.2f}%")
    print("-" * 30)

check_my_chance()