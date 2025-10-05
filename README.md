<<<<<<< HEAD
🩺 Health Insurance Charges Prediction API

This project is a Flask-based Machine Learning API that predicts medical insurance charges based on user details such as age, BMI, smoking status, and more.

It allows users to train, test, and predict using REST API endpoints — tested easily via Postman.

🚀 Features

/train → Train a Linear Regression model using uploaded CSV file

/test → Evaluate the trained model on test dataset

/predict → Predict insurance charges for new input data (JSON)

Auto-loads last saved model (model.pkl) if available

Fully compatible with Postman for testing

🧠 Tech Stack
Component	Technology Used
Language	Python 3
Framework	Flask
Machine Learning	scikit-learn
Data Handling	pandas, numpy
Model Storage	pickle
📂 Project Structure
Health_Insurance_API/
│
├── app.py                # Main Flask application
├── model.pkl             # Saved ML model (auto-created after training)
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── sample_data/
    ├── train.csv         # Sample training dataset
    └── test.csv          # Sample testing dataset

⚙️ Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/yourusername/Health_Insurance_API.git
cd Health_Insurance_API

2️⃣ Create a Virtual Environment
python -m venv venv
venv\Scripts\activate    # On Windows
source venv/bin/activate # On Mac/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Flask Application
python app.py


App will run at 👉 http://127.0.0.1:5000/

🧾 API Endpoints
🟢 1. /train → Train the Model

Method: POST
Body: Upload a CSV file using key file

✅ Example using Postman:

Type: form-data

Key: file

Value: select your CSV file

Response:

{
  "status": "Model trained and saved successfully"
}

🟣 2. /test → Test Model Accuracy

Method: POST
Body: Upload test dataset (CSV) with same structure

Response:

{
  "r2_score": 0.86,
  "mean_squared_error": 15432.76,
  "mean_absolute_error": 90.44
}

🔵 3. /predict → Predict Insurance Charges

Method: POST
Body Type: raw → JSON

Example Request:

{
  "age": 29,
  "bmi": 27.5,
  "children": 1,
  "sex": "female",
  "smoker": "no"
}


Example Response:

{
  "input": {
    "age": 29,
    "bmi": 27.5,
    "children": 1,
    "sex": "female",
    "smoker": "no"
  },
  "predicted_charges": 4023.58
}

🧩 Dataset Columns Explanation
Column	Description
age	Age of the person
sex	Gender (male/female)
bmi	Body Mass Index
children	Number of dependents
smoker	Whether the person smokes (yes/no)
region	Residential region (optional, dropped during preprocessing)
charges	Insurance charges (target column)
=======
# Health_Insurance_API
#MachineLearning #FlaskAPI #CodeSpyderTechnologies #Internship #PythonProject
>>>>>>> a733c3fdca77c995a05f36ff906f74c30c29bc0f
