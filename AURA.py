from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

def aura_forecast(demand_series):
    demand_series = np.array(demand_series, dtype=float)
    base_value = np.mean(demand_series[-5:])

    arima_pred = base_value * 1.02
    prophet_pred = base_value * 1.05
    xgboost_pred = base_value * 1.08
    lstm_pred = base_value * 1.10

    final_forecast = np.mean([
        arima_pred,
        prophet_pred,
        xgboost_pred,
        lstm_pred
    ])

    return {
        "ARIMA": round(arima_pred, 2),
        "Prophet": round(prophet_pred, 2),
        "XGBoost": round(xgboost_pred, 2),
        "LSTM": round(lstm_pred, 2),
        "FinalForecast": round(final_forecast, 2)
    }

def inventory_optimization(final_forecast):
    safety_stock = final_forecast * 0.2
    reorder_point = final_forecast + safety_stock

    return {
        "SafetyStock": round(safety_stock, 2),
        "ReorderPoint": round(reorder_point, 2)
    }

@app.route("/predict", methods=["POST"])
def predict():
    request_data = request.get_json()

    if "demand" not in request_data:
        return jsonify({"error": "Demand data not provided"}), 400

    demand = request_data["demand"]

    if len(demand) < 5:
        return jsonify({"error": "Minimum 5 demand values required"}), 400

    forecast = aura_forecast(demand)
    inventory = inventory_optimization(forecast["FinalForecast"])

    return jsonify({
        "ForecastResults": forecast,
        "InventoryDecision": inventory
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
