import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from flask import Flask, jsonify, request, render_template
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

model = pickle.load(open('xgb_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

import matplotlib.pyplot as plt
import io
import urllib, base64
import seaborn as sns
import matplotlib.pyplot as plt

def plot_distribution(data, x):
    plt.figure(figsize=(8,4))
    data[x].value_counts().plot(kind='pie', autopct='%1.0f%%', colors=['lightblue', 'lightgreen'])
    plt.title(f'Distribution of {x}')
    
    # Convert plot to PNG image
    pic_IObytes = io.BytesIO()
    plt.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())

    return "data:image/png;base64," + urllib.parse.quote(pic_hash)

# preprocess the input data
def apply_preprocessing(input_df, train=False, return_mapping=False, is_single_input=False):
    input_df['Gender'] = input_df['Gender'].astype('category')
    input_df['Gender'] = input_df['Gender'].cat.codes

    input_df['Active Member'] = input_df['Active Member'].astype('category')
    input_df['Active Member'] = input_df['Active Member'].cat.codes

    input_df['Credit Card'] = input_df['Credit Card'].astype('category')
    input_df['Credit Card'] = input_df['Credit Card'].cat.codes

    input_df['Balance'] = input_df['Balance'].astype('int64')
    input_df['EstimatedSalary'] = input_df['EstimatedSalary'].astype('int64')

    if train:
        input_df = input_df[(input_df['Age'] >= 18) & (input_df['Age'] <= 70)]

    if not is_single_input:
        input_df = pd.get_dummies(input_df, columns=['Geography'])

    num_features = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'Credit Card', 'Active Member', 'Geography_France', 'Geography_Germany', 'Geography_Spain']
    input_df[num_features] = scaler.transform(input_df[num_features])

    if return_mapping:
        original_indices = input_df.index
        return input_df, original_indices
    else:
        return input_df


@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'csvfile' not in request.files:
        return "No file part", 400

    csvfile = request.files['csvfile']
    if csvfile.filename == '':
        return "No selected file", 400

    # Read the uploaded CSV file
    input_df = pd.read_csv(csvfile.stream)
    input_df_original = input_df.copy() 

    # Preprocess the input data
    preprocessed_data, original_indices = apply_preprocessing(input_df, train=False, return_mapping=True)

    # Drop the 'Name' column
    preprocessed_data = preprocessed_data.drop('Name', axis=1)

    # Get predictions from the model
    predictions = model.predict(preprocessed_data)

    # Replace 0 with 'Not Churned' and 1 with 'Churned'
    predictions = np.where(predictions == 0, 'Not Churned', 'Churned')

    # Add a new column named 'Predicted Exited' to the input DataFrame
    input_df['Predicted Exited'] = pd.Series(predictions, index=original_indices)

    # Insert the 'Name' column from the original DataFrame to the end
    input_df['Name'] = input_df_original['Name']
    input_df = input_df[['Name', 'Predicted Exited', 'CreditScore', 'Age', 'EstimatedSalary', 'Tenure', 'Balance']]

    # Convert the updated input DataFrame to an HTML table
    result_table = input_df.to_html(index=False, border=0, classes=["table", "table-striped", "table-bordered", "table-hover"])

    return render_template('index.html', result_table=result_table)


@app.route('/visualize', methods=['POST'])
def visualize():
    csvfile = request.files['csvfile']
    if csvfile.filename == '':
        return "No selected file", 400

    # Read the uploaded CSV file
    input_df = pd.read_csv(csvfile.stream)

    # Preprocess the input data and get predictions
    preprocessed_data, original_indices = apply_preprocessing(input_df, train=False, return_mapping=True)

    # Drop the 'Name' column
    preprocessed_data = preprocessed_data.drop('Name', axis=1)

    # Get predictions from the model
    predictions = model.predict(preprocessed_data)

    # Replace 0 with 'Not Churned' and 1 with 'Churned'
    predictions = np.where(predictions == 0, 'Not Churned', 'Churned')
    input_df['Predicted Exited'] = pd.Series(predictions, index=original_indices)

    # Create the plot for Tenure
    plt.figure(figsize=(8,4))
    sns.histplot(data=input_df, x='Tenure', hue='Predicted Exited', multiple='stack', palette='Set2')
    plt.title('Distribution of Tenure by Predicted Exited')
    plt.tight_layout() 

    # Save the plot to a BytesIO object
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)

    # Convert the BytesIO object to a base64 string
    plot_url = base64.b64encode(bytes_image.getvalue()).decode()
    plot_url = 'data:image/png;base64,' + plot_url

    # Create the plot for NumOfProducts
    plt.figure(figsize=(8, 4))
    sns.countplot(data=input_df, x='NumOfProducts', hue='Predicted Exited', palette='Set3')
    plt.title('Number of Products by Predicted Exited')
    plt.tight_layout()

    # Save the plot to a BytesIO object
    bytes_image2 = io.BytesIO()
    plt.savefig(bytes_image2, format='png')
    bytes_image2.seek(0)

    def encode_geography(input_df):
        if 'Geography_France' in input_df.columns:
            input_df.loc[input_df['Geography_France'] == 1, 'Geography'] = 'France'
        if 'Geography_Germany' in input_df.columns:
            input_df.loc[input_df['Geography_Germany'] == 1, 'Geography'] = 'Germany'
        if 'Geography_Spain' in input_df.columns:
            input_df.loc[input_df['Geography_Spain'] == 1, 'Geography'] = 'Spain'
        return input_df


    # Create the plot for Geography
    input_df = encode_geography(input_df)
    geography_counts = input_df['Geography'].value_counts()
    colors = ['lightblue', 'lightgreen', 'pink']
    plt.figure(figsize=(8, 4))  # Adjust the size here
    plt.pie(geography_counts, labels=geography_counts.index, autopct='%1.1f%%', colors = colors)
    plt.title('Distribution of Customers by Geography')
    plt.tight_layout()

    # Save the plot to a BytesIO object
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)

    # Convert the BytesIO object to a base64 string
    plot_url4 = base64.b64encode(bytes_image.getvalue()).decode()
    plot_url4 = 'data:image/png;base64,' + plot_url4

    plot_url2 = base64.b64encode(bytes_image2.getvalue()).decode()
    plot_url2 = 'data:image/png;base64,' + plot_url2

    plot_url3 = plot_distribution(input_df, 'Predicted Exited')  

    return render_template('visualizations.html', plot_url=plot_url, plot_url2=plot_url2, plot_url3=plot_url3, plot_url4=plot_url4 )



@app.route('/predict_single', methods=['POST'])
def predict_single():
    float_features = [
        float(request.form['CreditScore']),
        float(request.form['Gender']),
        float(request.form['Age']),
        float(request.form['Tenure']),
        float(request.form['Balance']),
        float(request.form['NumOfProducts']),
        float(request.form['EstimatedSalary']),
        float(request.form['Credit Card']),
        float(request.form['Active Member']),
        float(request.form['Geography_France']),
        float(request.form['Geography_Germany']),
        float(request.form['Geography_Spain'])
    ]
    data = [float_features]

    # Preprocess the input data
    num_features = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'Credit Card', 'Active Member', 'Geography_France', 'Geography_Germany', 'Geography_Spain']
    
    preprocessed_data = apply_preprocessing(pd.DataFrame(data, columns=num_features), train=False, is_single_input=True) 

    prediction = model.predict(preprocessed_data)
    output = round(prediction[0], 2)

    if output == 0:
        prediction_text = 'Customer will not churn'
    else:
        prediction_text = 'Customer will churn'

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)

