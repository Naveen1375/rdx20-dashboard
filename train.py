import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

def generate_ra_dataset(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    
    l9_data = pd.DataFrame({
        "Spindle_Speed": [500, 600, 700, 600, 700, 500, 700, 500, 600],
        "Feed_Rate":     [120, 180, 240, 120, 180, 240, 120, 180, 240],
        "Depth_of_Cut":  [0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 1.0, 1.0, 1.0],
        "Ra":            [0.1696, 0.1717, 0.2397, 0.1643, 0.3264, 0.4374, 0.2611, 0.2859, 0.3308]
    })
    
    X_l9 = np.log(l9_data[["Spindle_Speed", "Feed_Rate", "Depth_of_Cut"]])
    y_l9 = np.log(l9_data["Ra"])
    
    lr_exact = LinearRegression()
    lr_exact.fit(X_l9, y_l9)
    
    spindle_speed = rng.uniform(400, 800, n)
    feed_rate     = rng.uniform(100, 260, n)
    depth_of_cut  = rng.uniform(0.3, 1.2, n)
    
    X_syn = np.log(pd.DataFrame({
        "Spindle_Speed": spindle_speed,
        "Feed_Rate":     feed_rate,
        "Depth_of_Cut":  depth_of_cut
    }))
    
    Ra_syn = np.exp(lr_exact.predict(X_syn))
    
    noise = rng.normal(1.0, 0.02, n)
    Ra_syn = np.clip(Ra_syn * noise, 0.081, 0.868)
    
    synth_df = pd.DataFrame({
        "Spindle_Speed": spindle_speed.round(1),
        "Feed_Rate":     feed_rate.round(1),
        "Depth_of_Cut":  depth_of_cut.round(3),
        "Ra":            Ra_syn.round(4)
    })
    
    l9_repeated = pd.concat([l9_data] * 10, ignore_index=True)
    return pd.concat([synth_df, l9_repeated], ignore_index=True)

df_ra = generate_ra_dataset()
X = df_ra[["Spindle_Speed", "Feed_Rate", "Depth_of_Cut"]]
y = df_ra["Ra"]

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

y_pred = model.predict(X_poly)

r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)

print("\n--- MODEL EVALUATION METRICS ---")
print(f"R2 Score: {r2:.4f}")
print(f"RMSE:     {rmse:.4f} um")
print(f"MAE:      {mae:.4f} um\n")

print("--- LEARNED POLYNOMIAL PARAMETERS (WEIGHTS) ---")
feature_names = poly.get_feature_names_out(["Speed", "Feed", "DoC"])
coefficients = model.coef_

print(f"w0 (Intercept) = {model.intercept_:.6f}")
for name, coef in zip(feature_names[1:], coefficients[1:]):
    print(f"w_{name.replace(' ', '*')} = {coef:.8f}")

joblib.dump((poly, model), "ra_prediction_model.pkl")
print("\nModel saved successfully locally as 'ra_prediction_model.pkl'")
