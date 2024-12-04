import pickle
import streamlit as st


# loading in the model to predict on the data
pickle_in = open('models/model.pkl', 'rb')
model = pickle.load(pickle_in)

# Load pre-trained LabelEncoders for categorical features
with open('models/encoders.pkl', 'rb') as encoder_file:
    encoders = pickle.load(encoder_file)


def page_style():
    return st.markdown(
        """
    <h1 style="text-align: center; font-family: sans-serif;">
    Income Prediction
    </h1>
    <hr>
    """,
        unsafe_allow_html=True
    )


def preprocess_features(age, working_class, education, marital_status,
                        occupation, relationship, race, gender,
                        capital_gain, capital_loss, hours_per_week, native_country):
    """
    Preprocess features: Encode categorical variables using pre-trained encoders.
    """
    # Encode categorical features
    working_class = encoders['workclass'].transform([working_class])[0]
    education = encoders['education'].transform([education])[0]
    marital_status = encoders['marital-status'].transform([marital_status])[0]
    occupation = encoders['occupation'].transform([occupation])[0]
    relationship = encoders['relationship'].transform([relationship])[0]
    race = encoders['race'].transform([race])[0]
    gender = encoders['gender'].transform([gender])[0]
    native_country = encoders['native-country'].transform([native_country])[0]

    return [age, int(working_class), int(education), int(marital_status), int(occupation),
            int(relationship), int(race), int(
                gender), capital_gain, capital_loss,
            hours_per_week, int(native_country)]


def predict(age, working_class,
            education, marital_status,
            occupation, relationship,
            race, gender,
            capital_gain, capital_loss,
            hours_per_week, native_country):
    """
    Predict income category based on user inputs.
    """
    # Preprocess the features
    feature_list = preprocess_features(age, working_class, education, marital_status,
                                       occupation, relationship, race, gender,
                                       capital_gain, capital_loss, hours_per_week, native_country)

    prediction = model.predict([feature_list])
    output = int(prediction[0])
    if output == 1:
        return "Employee Income is > 50K"
    else:
        return "Employee Income is <= 50K"


def main():
    page_style()

    age = st.number_input(label="Age", placeholder="Your Age",
                          value=None, min_value=0, max_value=100)
    working_class = st.selectbox(
        label="Working Class", index=0,
        options=("Federal-gov", "Local-gov", "Never-worked",
                 "Private", "Self-emp-inc", "Self-emp-not-inc",
                 "State-gov", "Without-pay"))
    education = st.selectbox(
        label="Education", index=0,
        options=("10th", "11th", "12th", "1st-4th", "5th-6th",
                 "7th-8th", "9th", "Assoc-acdm", "Assoc-voc",
                 "Bachelors", "Doctorate", "HS-grad", "Masters",
                 "Preschool", "Prof-school", "Some-college"))
    marital_status = st.selectbox(
        label="Marital Status", index=0,
        options=("Divorced", "Married-AF-spouse", "Married-civ-spouse",
                 "Married-spouse-absent", "Never-married", "Separated",
                 "Widowed"))
    occupation = st.selectbox(
        label="Occupation", index=0,
        options=("Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial",
                 "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct",
                 "Other-service", "Priv-house-serv", "Prof-specialty", "Protective-serv",
                 "Sales", "Tech-support", "Transport-moving"))
    relationship = st.selectbox(
        label="Relationship", index=0,
        options=("Husband", "Not-in-family", "Other-relative",
                 "Own-child", "Unmarried", "Wife"))
    race = st.selectbox(
        label="Race", index=0,
        options=("Amer-Indian-Eskimo", "Asian-Pac-Islander",
                 "Black", "Other", "White"))
    gender = st.selectbox(label="Gender", index=0,
                          options=("Male", "Female"))
    capital_gain = st.number_input(
        "Capital Gain", placeholder="[0-99999]", min_value=0, max_value=99999, value=None)
    capital_loss = st.number_input(
        "Capital Loss", placeholder="[0-4356]", min_value=0, max_value=4356, value=None)
    hours_per_week = st.number_input(
        "House Per Week", placeholder="[1-99]", min_value=0, max_value=99, value=None)
    native_country = st.selectbox(
        label="Native Country", index=0,
        options=("Cambodia", "Canada", "China",
                 "Columbia", "Cuba", "Dominican-Republic",
                 "Ecuador", "El-Salvador", "England",
                 "France", "Germany", "Greece",
                 "Guatemala", "Haiti", "Netherlands",
                 "Honduras", "HongKong", "Hungary",
                 "India", "Iran", "Ireland", "Italy",
                 "Jamaica", "Japan", "Laos", "Mexico",
                 "Nicaragua", "Outlying-US(Guam-USVI-etc)", "Peru",
                 "Philippines", "Poland", "Portugal", "Puerto-Rico",
                 "Scotland", "South", "Taiwan", "Thailand",
                 "Trinadad&Tobago", "United-States", "Vietnam", "Yugoslavia"))

    result = ""
    if st.button("Predict", use_container_width=True):
        result = predict(age, working_class,
                         education, marital_status,
                         occupation, relationship,
                         race, gender,
                         capital_gain, capital_loss,
                         hours_per_week, native_country)
    st.success(result)


if __name__ == '__main__':
    main()
