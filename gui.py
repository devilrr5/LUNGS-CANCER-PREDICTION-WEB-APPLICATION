import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

app = Flask(__name__)

# Load the trained machine learning model
model = pickle.load(open('svc_model.pkl', 'rb'))

# Load the scaler object used during training
scaler = StandardScaler()
scaler_file = 'scaler.pkl'
with open(scaler_file, 'rb') as f:
    scaler = pickle.load(f)

def convert_aqi_to_scaled(aqi):
    if aqi <= 50:
        return 0
    elif aqi <= 100:
        return 1
    elif aqi <= 200:
        return 2
    elif aqi <= 300:
        return 3 + (aqi - 200) / 100
    elif aqi <= 400:
        return 5 + (aqi - 300) / 100
    else:
        return 7 + (aqi - 400) / 100

def convert_level_to_scaled(level, levels_dict):
    return levels_dict[level]

def convert_alcohol_to_scaled(beer_volume, wine_volume, spirits_volume):
    beer_abv = 5
    wine_abv = 12
    spirits_abv = 40

    total_alcohol = beer_volume * (beer_abv / 100) + wine_volume * (wine_abv / 100) + spirits_volume * (spirits_abv / 100)

    if total_alcohol <= 40:
        return 1
    elif total_alcohol <= 100:
        return 2
    elif total_alcohol <= 150:
        return 3
    elif total_alcohol <= 250:
        return 4
    elif total_alcohol <= 500:
        return 5
    elif total_alcohol <= 750:
        return 6
    elif total_alcohol <= 1000:
        return 7
    else:
        return 8
@app.route('/')
def home():
    return render_template('home.html', favicon_url=url_for('static', filename='images/favicon.png'))

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        # Print out the form data for debugging
        print(request.form)

        # Extract the form data
        name = request.form.get('name')
        age = int(request.form.get('age'))
        gender = str(request.form.get('gender'))
        if gender.lower() == 'male': 
            gender = 1
        else: 
            gender = 2
        aqi = int(request.form.get('air_pollution'))
        air_pollution = convert_aqi_to_scaled(aqi)
        beer_volume = int(request.form.get('beer_volume'))
        wine_volume = int(request.form.get('wine_volume'))
        spirits_volume = int(request.form.get('spirits_volume'))
        alcohol_use = convert_alcohol_to_scaled(beer_volume, wine_volume, spirits_volume)

        new = 'dust_allergy_new' in request.form
        old = 'dust_allergy_old' in request.form
        road = 'dust_allergy_road' in request.form
        dust_allergy = 3 * (new + old + road)
        biological = 'occupational_hazards_biological' in request.form
        ergonomic = 'occupational_hazards_ergonomic' in request.form
        chemical = 'occupational_hazards_chemical' in request.form
        physical = 'occupational_hazards_physical' in request.form
        occupational_hazards = 2.5 * (biological + ergonomic + chemical + physical)
        genetic_risk = int(request.form.get('genetic_risk'))
        chronic_lung_disease = int(request.form.get('chronic_lung_disease'))
        balanced_diet = int(request.form.get('balanced_diet'))
        height = float(request.form.get('height'))
        weight = float(request.form.get('weight'))
        bmi = weight / (height ** 2)
        obesity = min(round((bmi - 18.5) / 3.25), 7)
        bidi = int(request.form.get('smoking_bidi'))
        cigar = int(request.form.get('smoking_cigar'))
        cigarette = int(request.form.get('smoking_cigarette'))
        electric_cigar = int(request.form.get('smoking_electric_cigar'))
        hookah = int(request.form.get('smoking_hookah'))
        smoking = (21*bidi + 30*cigarette + 10*electric_cigar + 14*cigar + hookah) / 30
        exposed_to_bidi = int(request.form.get('exposed_to_bidi', 0))
        exposed_to_cigar = int(request.form.get('exposed_to_cigar'))
        exposed_to_cigarette = int(request.form.get('exposed_to_cigarette'))
        exposed_to_electric_cigar = int(request.form.get('exposed_to_electric_cigar', 0))
        exposed_to_hookah = int(request.form.get('exposed_to_hookah'))
        passive_smoker = (21*exposed_to_bidi + 30*exposed_to_cigarette + 10*exposed_to_electric_cigar + 14*exposed_to_cigar + exposed_to_hookah) / 60
        chest_pain_level = request.form.get('chest_pain')
        chest_pain = convert_level_to_scaled(chest_pain_level, {
            "None": 0,
            "Very Mild (Occasional slight pain)": 1,
            "Mild (Pain when taking a deep breath)": 3,
            "Moderate (Persistent pain affecting daily activities)": 6,
            "Severe (Pain even at rest)": 8,
            "Life-threatening (Severe pain causing inability to move or breathe properly)": 9
        })
        coughing_blood_level = request.form.get('coughing_blood')
        coughing_blood = convert_level_to_scaled(coughing_blood_level, {
            "None": 0,
            "Very Mild (Streaks of blood)": 1,
            "Mild (Small amount)": 3,
            "Moderate (Noticeable amount)": 6,
            "Severe (Large amount)": 8,
            "Life-threatening (Profuse bleeding)": 9
        })
        fatigue_level = request.form.get('fatigue')
        fatigue = convert_level_to_scaled(fatigue_level, {
            "None": 0,
            "Very Mild (Slight tiredness after heavy exertion)": 1,
            "Mild (Tiredness after normal activity)": 3,
            "Moderate (Tiredness impacting daily activities)": 6,
            "Severe (Tiredness at rest)": 8,
            "Life-threatening (Extreme tiredness causing inability to move or function)": 9
        })
        weight_loss_level = request.form.get('weight_loss')
        weight_loss = convert_level_to_scaled(weight_loss_level, {
            "None": 0,
            "Very Mild (Occasional slight pain)": 1,
            "Mild (Pain when taking a deep breath)": 3,
            "Moderate (Persistent pain affecting daily activities)": 6,
            "Severe (Pain even at rest)": 8,
            "Life-threatening (Severe pain causing inability to move or breathe properly)": 9
        })
        shortness_of_breath_level = request.form.get('shortness_of_breath')
        shortness_of_breath = convert_level_to_scaled(shortness_of_breath_level, {
            "None": 0,
            "Mild (Slight difficulty on exertion)": 1,
            "Moderate (Difficulty after walking a few minutes)": 3,
            "Severe (Difficulty even at rest)": 6,
            "Very Severe (Unable to complete sentences without pausing for breath)": 8,
            "Life-threatening (Breathing stops for periods of time)": 9
        })
        wheezing_level = request.form.get('wheezing')
        wheezing = convert_level_to_scaled(wheezing_level, {
            "None": 0,
            "Very Mild (Occasional wheezing with exertion)": 1,
            "Mild (Wheezing a few times a week)": 2,
            "Moderate (Wheezing most days)": 4,
            "Severe (Wheezing multiple times a day)": 6,
            "Very Severe (Constant wheezing that interferes with daily activities)": 8
        })
        swallowing_difficulty_level = request.form.get('swallowing_difficulty')
        swallowing_difficulty = convert_level_to_scaled(swallowing_difficulty_level, {
            "None": 0,
            "Very Mild (Occasional difficulty swallowing)": 1,
            "Mild (Difficulty swallowing a few times a week)": 3,
            "Moderate (Difficulty swallowing most meals)": 6,
            "Severe (Difficulty swallowing every meal)": 8,
            "Very Severe (Unable to swallow, requiring alternative feeding methods)": 9
        })
        clubbing_level = request.form.get('clubbing')
        clubbing = convert_level_to_scaled(clubbing_level, {
            "None": 0,
            "Very Mild (Slight rounding of nails)": 1,
            "Mild (Rounding and enlargement of nails)": 3,
            "Moderate (Further rounding, enlargement and nail bed softening)": 6,
            "Severe (Large, bulbous, misshapen nails)": 8,
            "Life-threatening (Not applicable as this symptom is not life-threatening)": 9
        })
        frequent_cold_level = request.form.get('frequent_cold')
        frequent_cold = convert_level_to_scaled(frequent_cold_level, {
            "None (No colds in the past year)": 0,
            "Very Mild (1-2 colds in the past year)": 1,
            "Mild (3-4 colds in the past year)": 2,
            "Moderate (5-6 colds in the past year)": 4,
            "Severe (7-8 colds in the past year)": 6,
            "Very Severe (More than 8 colds in the past year)": 7
        })
        dry_cough_level = request.form.get('dry_cough')
        dry_cough = convert_level_to_scaled(dry_cough_level, {
            "None": 0,
            "Very Mild (Occasional coughing)": 1,
            "Mild (Coughing a few times a day)": 2,
            "Moderate (Coughing throughout the day)": 4,
            "Severe (Persistent coughing that interrupts daily activities)": 6,
            "Very Severe (Constant coughing that prevents sleep and normal activities)": 7
        })
        snoring_level = request.form.get('snoring')
        snoring = convert_level_to_scaled(snoring_level, {
            "None": 0,
            "Very Mild (Occasional snoring)": 1,
            "Mild (Snoring a few nights a week)": 2,
            "Moderate (Snoring most nights)": 4,
            "Severe (Snoring every night)": 6,
            "Very Severe (Loud snoring that can be heard in other rooms)": 7
        })

        data = np.zeros((23))
        data[0] = age
        data[1] = gender
        data[2] = air_pollution
        data[3] = alcohol_use
        data[4] = dust_allergy
        data[5] = occupational_hazards
        data[6] = genetic_risk
        data[7] = chronic_lung_disease
        data[8] = balanced_diet
        data[9] = obesity
        data[10] = smoking
        data[11] = passive_smoker
        data[12] = chest_pain
        data[13] = coughing_blood
        data[14] = fatigue
        data[15] = weight_loss
        data[16] = shortness_of_breath
        data[17] = wheezing
        data[18] = swallowing_difficulty
        data[19] = clubbing
        data[20] = frequent_cold
        data[21] = dry_cough
        data[22] = snoring

        # Convert the list to a numpy array with the desired shape
        new_data = np.array([data])

        # Standardize the new data using the loaded scaler
        new_data_scaled = scaler.transform(new_data)

        # Make predictions
        predictions = model.predict(new_data_scaled)

        # Determine the predicted outcome based on the prediction
        if predictions[0] <=0.5:
            outcome = 'Low'
        elif predictions[0] <=1.25:
            outcome = 'Medium'
        else:
            outcome = 'High'

        #chance of cancer
        percent=0
        percent= (predictions[0]/2)*100
        
        # Render the results template with the predicted outcome
        return render_template('results.html', outcome=outcome,name=name, age=age, gender=gender,percent=percent)
    else:
        # If the request method is not POST, redirect to the home page
        return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
