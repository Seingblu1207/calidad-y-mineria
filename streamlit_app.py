import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Predictor de Total fatalities",
    page_icon="✈️",
    layout="centered",
)

MODEL_PATH = Path(__file__).parent / "Notebooks" / "random_forest_model.pkl"

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(MODEL_PATH)

st.title("Predicción de Total fatalities")
st.write(
    "Usa el modelo Random Forest entrenado previamente para predecir la categoría de `Total fatalities` "
    "en función de los demás atributos del accidente."
)

st.markdown("---")

flight_phase_options = [
    "Flight",
    "Landing (descent or approach)",
    "Parking",
    "Takeoff (climb)",
    "Taxiing",
]

flight_type_options = [
    "Ambulance",
    "Calibration",
    "Cargo",
    "Charter/Taxi (Non Scheduled Revenue Flight)",
    "Delivery",
    "Demonstration",
    "Executive/Corporate/Business",
    "Ferry",
    "Fire fighting",
    "Geographical / Geophysical / Scientific",
    "Government",
    "Humanitarian",
    "Illegal (smuggling)",
    "Military",
    "Positioning",
    "Postal (mail)",
    "Private",
    "Scheduled Revenue Flight",
    "Skydiving / Paratroopers",
    "Spraying (Agricultural)",
    "Survey / Patrol / Reconnaissance",
    "Test",
    "Training",
]

crash_site_options = [
    "Airport (less than 10 km from airport)",
    "City",
    "Desert",
    "Mountains",
]

crash_cause_options = [
    "Human factor",
    "Other causes",
    "Technical failure",
    "Unknown",
    "Weather",
]

with st.form(key="prediction_form"):
    st.subheader("Datos de entrada")
    flight_phase = st.selectbox("Flight phase", flight_phase_options)
    flight_type = st.selectbox("Flight type", flight_type_options)
    crash_site = st.selectbox("Crash site", crash_site_options)
    crash_cause = st.selectbox("Crash cause", crash_cause_options)
    crew_on_board = st.number_input(
        "Crew on board",
        min_value=0.0,
        max_value=1000.0,
        value=2.0,
        step=1.0,
        format="%.0f",
    )
    pax_on_board = st.number_input(
        "Pax on board",
        min_value=0.0,
        max_value=1000.0,
        value=10.0,
        step=1.0,
        format="%.0f",
    )
    submit_button = st.form_submit_button("Predecir Total fatalities")

if submit_button:
    input_data = pd.DataFrame([
        {
            "Flight phase": flight_phase,
            "Flight type": flight_type,
            "Crash site": crash_site,
            "Crew on board": crew_on_board,
            "Pax on board": pax_on_board,
            "Crash cause": crash_cause,
        }
    ])

    prediction = model.predict(input_data)[0]
    st.success(f"Predicción: {prediction}")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0]
        classes = model.classes_
        proba_df = pd.DataFrame({"Total fatalities": classes, "Probabilidad": proba})
        proba_df = proba_df.sort_values("Probabilidad", ascending=False).reset_index(drop=True)
        st.markdown("### Probabilidades por categoría")
        st.dataframe(proba_df.style.format({"Probabilidad": "{:.2%}"}))

st.markdown("---")
st.write("**Notas:**")
st.write(
    "El modelo carga el Random Forest exportado en `Notebooks/random_forest_model.pkl`. "
    "Asegúrate de ejecutar la app desde la carpeta del proyecto para que la ruta relativa funcione correctamente."
)
