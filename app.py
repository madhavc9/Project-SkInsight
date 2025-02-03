import sqlite3
from flask import Flask,jsonify, render_template, request, redirect, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from roboflow import Roboflow
import json
import supervision as sv
import os
import uuid  # For generating unique filenames
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import cv2
import pandas as pd
from joblib import load

df=pd.read_csv(r"dataset/updated_skincare_products.csv")

app = Flask(__name__)
app.secret_key = '4545'
DATABASE = 'app.db'

def create_tables():
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          username TEXT UNIQUE NOT NULL,
                          password TEXT NOT NULL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS survey_responses (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          user_id INTEGER NOT NULL,
                          name TEXT NOT NULL,
                          age TEXT NOT NULL,
                          gender TEXT NOT NULL,
                          concerns TEXT NOT NULL,
                          acne_frequency TEXT NOT NULL,
                          comedones_count TEXT NOT NULL,
                          first_concern TEXT NOT NULL,
                          cosmetic_usage TEXT NOT NULL,
                          skin_reaction TEXT NOT NULL,
                          skin_type TEXT NOT NULL,
                          medications TEXT NOT NULL,
                          skincare_routine TEXT NOT NULL,
                          stress_level TEXT NOT NULL,
                          FOREIGN KEY (user_id) REFERENCES users(id))''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS appointment( 
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       name TEXT ,
                       email TEXT ,
                       date TEXT, 
                       skin TEXT,
                       phone TEXT,
                       age TEXT,
                       address TEXT, 
                       status BOOLEAN,
                       username TEXT
                       )''')

def insert_user(username, password):
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        connection.commit()
        
def insert_appointment_data(name,email,date,skin,phone,age,address,status,username):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute('''INSERT INTO appointment (name,email,date,skin,phone,age,address,status,username)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?,?)''', ( name,email,date,skin,phone,age,address,status,username))
    conn.commit()
    conn.close()                    

# Load the trained model
loaded_model = load("model/final_model.h5") 
#product recommend model trained on kaggle bcz there GPU

def findappointment(user):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute("""SELECT * FROM appointment WHERE username = ?  """,(user,))
    conn.commit()
    users=c.fetchall()
    conn.close()    
    return users

def findallappointment():
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute("""SELECT * FROM appointment  """,)
    conn.commit()
    users=c.fetchall()
    conn.close()    
    return users
    

def get_user(username):
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        return cursor.fetchone()

def insert_survey_response(user_id, name, age, gender, concerns, acne_frequency, comedones_count, first_concern, cosmetic_usage, skin_reaction, skin_type, medications,skincare_routine,stress_level):
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute("INSERT INTO survey_responses (user_id, name, age, gender, concerns, acne_frequency, comedones_count, first_concern, cosmetic_usage, skin_reaction, skin_type, medications,skincare_routine,stress_level) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                       (user_id, name, age, gender, concerns, acne_frequency, comedones_count, first_concern, cosmetic_usage, skin_reaction, skin_type, medications,skincare_routine,stress_level))
        connection.commit()

def init_app():
    create_tables()

# Skin detection model initialization
# this on roboflow trained
rf_skin = Roboflow(api_key="8RSJzoEweFB7NxxNK6fg")
project_skin = rf_skin.workspace().project("skin-detection-pfmbg")
model_skin = project_skin.version(2).model

# Oilyness detection model initialization
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Gqf1hrF7jdAh8EsbOoTM"
)

# Store unique classes
unique_classes = set()

# Mapping for oily skin class
class_mapping = {
    "Jenis Kulit Wajah - v6 2023-06-17 11-53am": "oily skin",
    "-": "normal/dry skin"  
}
def recommend_products_based_on_classes(classes):
    recommendations = []
    # Convert DataFrame column names to lower case for case-insensitive comparison
    df_columns_lower = [column.lower() for column in df.columns]
    for skin_condition in classes:
        # Convert each class to lower case to ensure case-insensitive comparison
        skin_condition_lower = skin_condition.lower()
        if skin_condition_lower in df_columns_lower:
            # Find the original column name that matches the lower case version
            original_column = df.columns[df_columns_lower.index(skin_condition_lower)]
            filtered_products = df[df[original_column] == 1][['Brand', 'Name', 'Price', 'Ingredients']]
            
            # Modify the ingredients to include only the first five
            filtered_products['Ingredients'] = filtered_products['Ingredients'].apply(lambda x: ', '.join(x.split(', ')[:5]))
            
            # Convert DataFrame to a list of dictionaries
            products_list = filtered_products.head(5).to_dict(orient='records') # Show top 5 recommendations
            recommendations.append((skin_condition, products_list))
        else:
            print(f"Warning: No column found for skin condition '{skin_condition}'")
    return recommendations

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        # Process image and get predictions
        image_file = request.files['image']
        
        # Generate a unique filename for the image
        image_filename = str(uuid.uuid4()) + '.jpg'
        image_path = os.path.join('static', image_filename)
        
        # Save the uploaded image
        image_file.save(image_path)

        # Skin detection
        skin_result = model_skin.predict(image_path, confidence=15, overlap=30).json()
        skin_labels = [item["class"] for item in skin_result["predictions"]]
        for label in skin_labels:
            unique_classes.add(label)

        # Oilyness detection with confidence threshold
        custom_configuration = InferenceConfiguration(confidence_threshold=0.3)
        with CLIENT.use_configuration(custom_configuration):
            oilyness_result = CLIENT.infer(image_path, model_id="oilyness-detection-kgsxz/1")
        
        # Check if oilyness prediction is empty
        if not oilyness_result['predictions']:
            unique_classes.add("dryness")
        else:
            oilyness_classes = [class_mapping.get(prediction['class'], prediction['class']) for prediction in oilyness_result['predictions'] if prediction['confidence'] >= 0.3]
            for label in oilyness_classes:
                unique_classes.add(label)

        # Draw boxes on the image
        image = cv2.imread(image_path)
        detections = sv.Detections.from_inference(skin_result)
        label_annotator = sv.LabelAnnotator()
        bounding_box_annotator = sv.BoxAnnotator()

        annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        recommended_products = recommend_products_based_on_classes(list(unique_classes))
        prediction_data = {
            'classes': list(unique_classes),
            'recommendations': recommended_products
        }
        print(prediction_data)
        # Save the annotated image as annotations_0.jpg
        annotated_image_path = os.path.join('static', 'annotations_0.jpg')
        cv2.imwrite(annotated_image_path, annotated_image)

        return render_template('face_analysis.html', data=prediction_data)
    else:
        return render_template('face_analysis.html', data=[])

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = request.form['name']
        age = request.form['age']  # Assuming you have an age field in the registration form
        hashed_password = generate_password_hash(password)

        if get_user(username):
            return "Username already exists. Please choose a different one."

        insert_user(username, hashed_password)
        # Store name and age in session
        session['name'] = name
        session['age'] = age
        
        return redirect('/')

    return render_template('register.html')

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = get_user(username)
        if user and check_password_hash(user[2], password):
            session['username'] = username
            user_id = user[0]
            if username=='doctor1':
                return redirect(url_for('allappoint'))
            else :
                survey_response = get_survey_response(user_id)
                if survey_response:
                    return redirect('/profile')
                else:
                    # If the user hasn't completed the survey, redirect them to the survey page
                    return redirect('/survey')

        return "Invalid username or password"

    return render_template('login.html')

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    if request.method == 'POST':
        user_id = get_user(session['username'])[0]
        name = session.get('name', '')
        age = session.get('age', '')
        gender = request.form['gender']
        concerns = ",".join(request.form.getlist('concerns'))
        acne_frequency = request.form['acne_frequency']
        comedones_count = request.form['comedones_count']
        first_concern = request.form['first_concern']
        cosmetics_usage = request.form['cosmetics_usage']
        skin_reaction = request.form['skin_reaction']
        skin_type = request.form['skin_type_details']
        medications = request.form['medications']
        skincare_routine = request.form['skincare_routine']
        stress_level = request.form['stress_level']

        insert_survey_response(user_id, name, age, gender, concerns, acne_frequency, comedones_count, first_concern, cosmetics_usage, skin_reaction, skin_type, medications, skincare_routine, stress_level)
        return redirect(url_for('profile'))

    return render_template('survey.html', name=session.get('name'), age=session.get('age'), occupation=session.get('occupation'))


def get_survey_response(user_id):
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM survey_responses WHERE user_id = ?", (user_id,))
        return cursor.fetchone()  
    
def update_appointment_status(appointment_id):
    conn = sqlite3.connect('app.db')
    
    c = conn.cursor()
    c.execute("""UPDATE appointment SET status = ? WHERE id = ?""", (True, appointment_id))
    conn.commit()
    
    conn.close()    

@app.route('/profile')
def profile():
    if 'username' in session:
        user_id = get_user(session['username'])[0]
        survey_response = get_survey_response(user_id)
        if survey_response:
            return render_template('profile.html', 
                                   name=survey_response[2],
                                   age=survey_response[3],
                                   gender=survey_response[4],
                                   concerns=survey_response[5],
                                   acne_frequency=survey_response[6],
                                   comedones_count=survey_response[7],
                                   first_concern=survey_response[8],
                                   cosmetics_usage=survey_response[9],
                                   skin_reaction=survey_response[10],
                                   skin_type_details=survey_response[11],
                                   medications=survey_response[12],
                                   skincare_routine=survey_response[13],
                                   stress_level=survey_response[14],
                                #   monthly_spending=survey_response[15],
                                  # technology_improvement=survey_response[16],
                                  # heard_about_AI=survey_response[17],
                                  )
    return redirect('/')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/')

@app.route('/bookappointment')
def bookappointment():
   return render_template('bookappointment.html')

@app.route("/appointment",methods=["POST"])
def appointment():
    name=request.form.get("name")
    email=request.form.get("email")
    date=request.form.get("date")
    skin=request.form.get("skin")
    phone=request.form.get("phone")
    age=request.form.get("age")
    address=request.form.get("address")
    username=session['username']
    status=False
    print('hello')
    insert_appointment_data(name,email,date,skin,phone,age,address,status,username)
    return redirect(url_for('bookappointment'))

@app.route("/allappointments")
def allappoint():
    all_appointments=findallappointment()
    print("Appointments fetched successfully")
    print(json.dumps(all_appointments))
    return render_template('doctor.html',appointments=json.dumps(all_appointments))


@app.route("/userappointment")
def userappoint():
    user = session['username']
    print(user)
    all_appointments = findappointment(user)
    print("Appointments fetched successfully")
    print(json.dumps(all_appointments))
    return render_template('userappointment.html',all_appointments=json.dumps(all_appointments))

@app.route("/update_status", methods=["POST"])
def update_status():
    if request.method == "POST":
        # Get the appointment ID and status from the request
        appointment_id = request.form.get("appointment_id")
        type=request.form.get("type")
        print(appointment_id)
        
        update_appointment_status(appointment_id)
        print("appointment_id")
      
        return "updated"
@app.route("/doctor")
def doctor():
        
        return render_template('doctor.html') 

@app.route("/delete_user_request", methods=["POST"])
def delete_user_request():
    if request.method == "POST":
        # Get the appointment ID and status from the request
        id = request.form.get("id")
        conn = sqlite3.connect('app.db')
        c = conn.cursor()
        c.execute("""DELETE FROM appointment WHERE id = ?""", (id,))
        conn.commit()
        users=c.fetchall()
        conn.close()  
        return "deleted successfully"     

if __name__ == '__main__':
    init_app()
    app.run(debug=True)
