import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Helpers (NumPy only)
# -----------------------------
def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)

def make_sequences(series_1d, window):
    """Return X (N, window) and y (N,) from 1D array."""
    series_1d = np.asarray(series_1d, dtype=np.float64).reshape(-1)
    X, y = [], []
    for i in range(len(series_1d) - window):
        X.append(series_1d[i:i + window])
        y.append(series_1d[i + window])
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)

# -----------------------------
# Loaded RNN (same as notebook save format)
# -----------------------------
class LoadedRNN:
    def __init__(self, npz_bytes_or_path):
        data = np.load(npz_bytes_or_path, allow_pickle=True)

        # weights
        self.Wxh = data["Wxh"]
        self.Whh = data["Whh"]
        self.bh  = data["bh"]
        self.Why = data["Why"]
        self.by  = data["by"]

        # scaler info
        self.data_min = float(data["data_min"])
        self.data_max = float(data["data_max"])
        self.n_input  = int(data["n_input"])

        denom = (self.data_max - self.data_min)
        self._denom = denom if denom != 0 else 1.0

        # hidden size sanity
        self.hidden_size = int(self.Whh.shape[0])

    def scale(self, x):
        x = np.asarray(x, dtype=np.float64)
        return (x - self.data_min) / self._denom

    def inverse_scale(self, x_scaled):
        x_scaled = np.asarray(x_scaled, dtype=np.float64)
        return x_scaled * self._denom + self.data_min

    def predict_next_from_window(self, last_values_original_units):
        """
        last_values_original_units: list/array length n_input in ORIGINAL units
        returns: next value prediction in ORIGINAL units
        """
        x = np.asarray(last_values_original_units, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.n_input:
            raise ValueError(f"Need exactly {self.n_input} values, got {x.shape[0]}.")

        x_scaled = self.scale(x).reshape(self.n_input, 1)  # (T, 1)

        h = np.zeros((self.hidden_size, 1), dtype=np.float64)
        for t in range(self.n_input):
            x_t = x_scaled[t].reshape(1, 1)  # (1,1)
            h = np.tanh(self.Wxh @ x_t + self.Whh @ h + self.bh)

        y_hat_scaled = (self.Why @ h + self.by).squeeze()
        y_hat = self.inverse_scale(y_hat_scaled)
        return float(np.asarray(y_hat).squeeze())

    def forecast(self, seed_window_original_units, steps):
        """
        seed_window_original_units: length n_input
        steps: int
        returns: list of length steps predictions in ORIGINAL units
        """
        history = list(np.asarray(seed_window_original_units, dtype=np.float64).reshape(-1))
        preds = []
        for _ in range(int(steps)):
            nxt = self.predict_next_from_window(history[-self.n_input:])
            preds.append(nxt)
            history.append(nxt)
        return preds

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Milk Production Forecast (NumPy RNN)", layout="wide")
st.title("🥛 Milk Production Forecast (Memory RNN) — NumPy Only")

st.write(
    "This app loads a **trained NumPy RNN** (saved as `.npz`) and makes time-series predictions.\n\n"
)

# Defaults (match your notebook paths)
DEFAULT_CSV = "monthly_milk_production.csv"
DEFAULT_NPZ = "rnn_milk_model.npz"

with st.sidebar:
    st.header("📦 Load files")

    st.caption("Option A: Put files in the same folder and the app will auto-detect.\n"
               "Option B: Upload them here.")

    uploaded_npz = st.file_uploader("Upload model (.npz)", type=["npz"])
    uploaded_csv = st.file_uploader("Upload dataset (.csv)", type=["csv"])

    st.divider()
    st.header("🔧 Settings")
    horizon = st.slider("How many months to predict ahead?", min_value=1, max_value=36, value=12)

# -------- Load model ----------
model = None
model_source = None

if uploaded_npz is not None:
    try:
        model = LoadedRNN(uploaded_npz)
        model_source = "uploaded"
    except Exception as e:
        st.error(f"Could not load uploaded .npz model: {e}")
        st.stop()
else:
    if os.path.exists(DEFAULT_NPZ):
        try:
            model = LoadedRNN(DEFAULT_NPZ)
            model_source = "local"
        except Exception as e:
            st.error(f"Found {DEFAULT_NPZ} but failed to load it: {e}")
            st.stop()

if model is None:
    st.warning(
        "I can't find your trained model file.\n\n"
        "✅ Fix: Run your notebook cell that saves the model as **rnn_milk_model.npz**, "
        "OR upload the `.npz` file in the sidebar."
    )
    st.stop()

st.success(f"Loaded model ✅ (source: {model_source}). Window size (n_input) = {model.n_input}")

# -------- Load dataset ----------
df = None
if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
        st.stop()
else:
    if os.path.exists(DEFAULT_CSV):
        df = pd.read_csv(DEFAULT_CSV)
    else:
        st.warning(
            "I can't find the dataset CSV.\n\n"
            "✅ Fix: Put **monthly_milk_production.csv** in the same folder OR upload it in the sidebar."
        )
        st.stop()

# Validate columns
expected_cols = {"Date", "Production"}
if not expected_cols.issubset(set(df.columns)):
    st.error(f"CSV must have columns {expected_cols}. Found: {list(df.columns)}")
    st.stop()

# Clean + sort
df = df.copy()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date")
df["Production"] = pd.to_numeric(df["Production"], errors="coerce")
df = df.dropna(subset=["Production"]).reset_index(drop=True)

series = df["Production"].to_numpy(dtype=np.float64)

if len(series) <= model.n_input + 2:
    st.error("Dataset is too short for this window size.")
    st.stop()

# Split train/test (simple)
split_ratio = 0.80
split_idx = int(len(series) * split_ratio)
if split_idx <= model.n_input:
    split_idx = model.n_input + 1

train = series[:split_idx]
test = series[split_idx:]

st.subheader("📊 Dataset preview")
st.dataframe(df.head(10), use_container_width=True)

# -------- Evaluate on test (walk-forward) ----------
# Make predictions for test points using last window from history
history = list(train)
test_preds = []

for i in range(len(test)):
    window = history[-model.n_input:]
    pred = model.predict_next_from_window(window)
    test_preds.append(pred)
    # add the REAL value (so next step uses real history)
    history.append(test[i])

test_preds = np.array(test_preds, dtype=np.float64)

col1, col2, col3 = st.columns(3)
col1.metric("MAE (test)", f"{mae(test, test_preds):.3f}")
col2.metric("RMSE (test)", f"{rmse(test, test_preds):.3f}")
col3.metric("R² (test)", f"{r2_score(test, test_preds):.3f}")

# -------- Forecast future ----------
seed_window = series[-model.n_input:]
future_preds = model.forecast(seed_window, horizon)

# Build future dates (monthly)
last_date = df["Date"].iloc[-1]
future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

# -------- Plot ----------
st.subheader("📈 Plot: Actual vs Predicted")

fig = plt.figure()
plt.plot(df["Date"].to_numpy(), series, label="Actual")
plt.plot(df["Date"].iloc[split_idx:].to_numpy(), test_preds, label="Test predictions")
plt.plot(future_dates.to_numpy(), np.array(future_preds), label="Future forecast")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Milk Production")
st.pyplot(fig, clear_figure=True)

# -------- Table output ----------
st.subheader("🧾 Future forecast values")
out_df = pd.DataFrame({"Date": future_dates, "Predicted_Production": future_preds})
st.dataframe(out_df, use_container_width=True)

# -------- Manual prediction mode ----------
st.subheader("🎮 Try your own window (manual input)")
st.write(
    f"Give me **{model.n_input} numbers** (latest to oldest doesn’t matter as long as you keep the same order each time). "
    "The model will guess the next month."
)

default_text = ", ".join([str(int(x)) for x in seed_window])
user_text = st.text_area("Enter comma-separated values:", value=default_text, height=80)

try:
    vals = [float(v.strip()) for v in user_text.split(",") if v.strip() != ""]
    if len(vals) != model.n_input:
        st.info(f"Please enter exactly {model.n_input} values. You entered {len(vals)}.")
    else:
        next_pred = model.predict_next_from_window(vals)
        st.success(f"✅ Predicted next month production: {next_pred:.3f}")
except Exception as e:
    st.error(f"Input problem: {e}")