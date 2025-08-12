from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import shap
import numpy as np
import uvicorn
import pandas as pd

# To initialize app
app = FastAPI(title="Fake Job Detection API")

# Loading model and vectorizer
model = joblib.load("xgboost_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# To get feature names from vectorizer
feature_names = vectorizer.get_feature_names_out()

# To create SHAP explainer using feature names
explainer = shap.Explainer(model, feature_names=feature_names)

# Defining request part
class JobInput(BaseModel):
    text: str

# We use the first 1000 words (or fewer) as the background
import numpy as np
background_data = np.zeros((1, len(vectorizer.get_feature_names_out())))

# Creating masker and explainer using model.predict
masker = shap.maskers.Independent(background_data)
explainer = shap.Explainer(model.predict, masker=masker, feature_names=vectorizer.get_feature_names_out())


# Root endpoint
@app.post("/predict/")
def predict_fake_job(input_data: JobInput):
    import traceback
    try:
        if not input_data.text:
            raise HTTPException(status_code=400, detail="Text input is required.")

        # Vectorize input and convert to dense
        X_vec_sparse = vectorizer.transform([input_data.text])
        X_vec_dense = X_vec_sparse.toarray()

        # Print shape for debugging
        print("Vectorized shape:", X_vec_dense.shape)

        # Make prediction
        prediction = model.predict(X_vec_dense)[0]
        proba = model.predict_proba(X_vec_dense)[0][1]

        # SHAP explanation
        print("Explaining with SHAP...")
        shap_values = explainer(X_vec_dense)
        print("SHAP values:", shap_values)

        instance_values = shap_values[0]

        top_features = sorted(
            zip(instance_values.values, instance_values.data, explainer.feature_names),
            key=lambda x: abs(x[0]),
            reverse=True
        )[:5]

        explanation = [
            {"feature": f, "value": str(v), "shap_impact": float(s)}
            for s, v, f in top_features
        ]

        return {
            "prediction": int(prediction),
            "probability_fraudulent": round(float(proba), 3),
            "top_shap_features": explanation
        }

    except Exception as e:
        print("❌ ERROR OCCURRED:")
        traceback.print_exc()  # This will show the full traceback in my terminal
        raise HTTPException(status_code=500, detail=str(e))
