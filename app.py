from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load("models/attrition_real_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    income = float(request.form["income"])
    satisfaction = float(request.form["satisfaction"])
    years = float(request.form["years"])
    overtime = float(request.form["overtime"])
    distance = float(request.form["distance"])

    employee_data = [[
        age,
        income,
        satisfaction,
        years,
        overtime,
        distance
    ]]

    prediction = model.predict(employee_data)[0]

    if prediction == 1:
        result = "Employee is likely to leave the company."
    else:
        result = "Employee is likely to stay with the company."

    return render_template(
        "index.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)
