from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import torch
import matplotlib.pyplot as plt
from timeseries_script import normalize, denormalize, predict_sequence, LSTMModel
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        receipt_counts = []

        for i in range(1, 13):
            receipt_counts.append(float(request.form.get(f'receipt_{i}')))

        num_days = int(request.form["num_days"])

        model = LSTMModel().to(device="cuda")
        model.load_state_dict(torch.load("timeseries_predictor2.pth"))

        predictions = predict_sequence(model, receipt_counts, num_days)

        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()

        predictions_int = [int(num) for num in predictions]

        img = generate_plot(receipt_counts, predictions_int)

        return render_template("results.html", predictions=predictions_int, plot_url=img)

    return render_template("index.html")

def generate_plot(receipt_counts, predictions):
    total_days = receipt_counts + predictions

    fig, ax = plt.subplots()
    ax.plot(range(1, len(receipt_counts) + 1), receipt_counts, label='Last 12 Days', color='blue', marker='o')
    ax.plot(range(len(receipt_counts) + 1, len(total_days) + 1), predictions, label='Predicted Days', color='orange', marker='x')

    ax.set_xlabel('Days')
    ax.set_ylabel('Receipt Counts (in millions)')
    ax.set_title('Receipt Counts Over Time')
    ax.legend()

    img = io.BytesIO()
    FigureCanvas(fig).print_png(img)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)  
    return f"data:image/png;base64,{plot_url}"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)