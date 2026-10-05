import streamlit as st
import joblib
from scipy.sparse import hstack


# ----------------------------------------------------
# LOAD TRAINED MODELS
# ----------------------------------------------------

model = joblib.load("difficulty_model_logistic.pkl")
vectorizer = joblib.load("vectorizer.pkl")
mlb = joblib.load("tag_encoder.pkl")


# ----------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------=

st.set_page_config(
    page_title="LeetCode Difficulty Predictor",
    page_icon="Difficulty Predictor",
    layout="centered"
)


# ----------------------------------------------------
# TITLE
# ----------------------------------------------------
st.title("🧠 LeetCode Difficulty Predictor")

st.write(
    "Enter a LeetCode problem and predict whether it is "
    "Easy, Medium, or Hard."
)


# ----------------------------------------------------
# INPUT
# ----------------------------------------------------

question = st.text_area(
    "Problem Description",
    height=250,
    placeholder="Paste the LeetCode problem description here..."
)


tags_input = st.text_input(
    "Problem Tags",
    placeholder="Example: Array, Hash Table, Dynamic Programming"
)


# ----------------------------------------------------
# PREDICTION
# ----------------------------------------------------
if st.button("Predict Difficulty", type="primary"):

    if question.strip() == "":
        st.warning("Please enter a problem description.")

    else:

        # ----------------------------------------------------
        # TF-IDF
        # ----------------------------------------------------

        question_tfidf = vectorizer.transform(
            [question]
        )


        # ----------------------------------------------------
        # TAGS
        # ----------------------------------------------------

        tags = [
            tag.strip()
            for tag in tags_input.split(",")
            if tag.strip()
        ]

        question_tags = mlb.transform([tags])


        # ----------------------------------------------------
        # COMBINE FEATURES
        # ----------------------------------------------------

        X_new = hstack([
            question_tfidf,
            question_tags
        ])


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(X_new)[0]


        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        probabilities = model.predict_proba(X_new)[0]

        class_probabilities = dict(
            zip(
                model.classes_,
                probabilities
            )
        )

        confidence = max(probabilities)


        #-----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.subheader("Prediction")

        if prediction == "Easy":

            st.success(
                f"🟢 Difficulty: {prediction}"
            )

        elif prediction == "Medium":

            st.warning(
                f"🟡 Difficulty: {prediction}"
            )

        else:

            st.error(
                f"🔴 Difficulty: {prediction}"
            )


        #------------------------------------------------------
        # CONFIDENCE
        #------------------------------------------------------

        st.subheader("Model Confidence")

        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )

        #------------------------------------------------------
        # CLASS PROBABILITIES
        #------------------------------------------------------
        st.subheader("Class Probabilities")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Easy",
                f"{class_probabilities.get('Easy', 0):.2%}"
            )

        with col2:

            st.metric(
                "Medium",
                f"{class_probabilities.get('Medium', 0):.2%}"
            )

        with col3:

            st.metric(
                "Hard",
                f"{class_probabilities.get('Hard', 0):.2%}"
            )


        # ----------------------------------------------------
        # PROBABILITY BAR
        # ----------------------------------------------------
        st.subheader("Probability Distribution")

        st.progress(
            float(
                class_probabilities.get("Easy", 0)
            ),
            text="Easy"
        )

        st.progress(
            float(
                class_probabilities.get("Medium", 0)
            ),
            text="Medium"
        )

        st.progress(
            float(
                class_probabilities.get("Hard", 0)
            ),
            text="Hard"
        )


# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------

st.divider()

st.caption(
    "AI-based LeetCode Difficulty Prediction System"
)