import optuna
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.models import Sequential
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import plotly.offline as pyo
import optuna.visualization as ov
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Layer
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt



np.random.seed(42)
tf.random.set_seed(42)

# Load and preprocess your data
data = pd.read_csv('surv_data.csv')
X = data.drop(['patient_id', 'time_to_event', 'event_occurred'], axis=1)
y = data[['time_to_event', 'event_occurred']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_data = pd.DataFrame(X_test)
Y_data = pd.DataFrame(y_test)
data1 = pd.merge(X_data, Y_data, left_index=True, right_index=True)
print(data1)


class CoxPHLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.beta = None

    def build(self, input_shape):
        self.beta = self.add_weight(name='beta', shape=(input_shape[-1], 1),
                                    initializer='zeros', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.beta)


def cox_loss(y_true, y_pred):
    """
    y_true: (n, 2) tensor with true survival times and event indicators.
    y_pred: (n, 1) tensor with predicted risk scores.
    """
    # Splitting survival times and event indicators
    survival_times = y_true[:, 0]
    event_indicators = y_true[:, 1]

    # Sorting by survival time, descending
    sorted_indices = tf.argsort(survival_times, direction='DESCENDING')
    sorted_survival_times = tf.gather(survival_times, sorted_indices)
    sorted_event_indicators = tf.gather(event_indicators, sorted_indices)
    sorted_risk_scores = tf.gather(y_pred[:, 0], sorted_indices)

    # Calculating risk set for each time
    risk_set = tf.cast(tf.greater_equal(tf.expand_dims(sorted_survival_times, axis=-1), sorted_survival_times), dtype=tf.float32)

    # Calculating the numerator (exp(risk score) for events)
    exp_risk_scores = tf.exp(sorted_risk_scores)
    numerator = tf.where(sorted_event_indicators == 1, exp_risk_scores, 0.0)

    # Calculating the denominator (sum of exp(risk scores) for risk set)
    denominator = tf.reduce_sum(tf.multiply(exp_risk_scores, risk_set), axis=1)
    
    # Cox partial likelihood
    partial_likelihood = tf.divide(numerator, denominator + tf.keras.backend.epsilon())
    partial_likelihood = tf.where(partial_likelihood > 0, partial_likelihood, tf.keras.backend.epsilon())

    # Negative log likelihood
    negative_log_likelihood = -tf.reduce_sum(tf.math.log(partial_likelihood))

    return negative_log_likelihood



def objective(trial):
    # Suggest values for the hyperparameters
    encoding_dim = trial.suggest_categorical('encoding_dim', [32, 64, 128])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    # Building the autoencoder
    input_layer = Input(shape=(X_train.shape[1],))
    encoded = Dense(encoding_dim, activation="relu")(input_layer)
    encoded = Dropout(dropout_rate)(encoded)
    decoded = Dense(X_train.shape[1], activation="sigmoid")(encoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)

    # Compile the autoencoder
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')

    # Train the autoencoder
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, verbose=0)

    # Extracting the encoded features
    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test)

    # Building the Cox model
    model = Sequential()
    model.add(Input(shape=(encoding_dim,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate / 2))
    model.add(Dense(32, activation='relu'))
    model.add(CoxPHLayer())
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=cox_loss)

    # Train the Cox model
    model.fit(X_train_encoded, y_train, epochs=50, batch_size=32, verbose=0)

    # Predict on validation set and calculate concordance index
    predictions = model.predict(X_test_encoded)
    predicted_risk = model.predict(X_test_encoded)
    median_risk = np.median(predicted_risk)

# Stratify patients based on the median risk score
    risk_stratification = ['High Risk' if risk > median_risk else 'Low Risk' for risk in predicted_risk.flatten()]

# Add the risk stratification to your dataframe for further analysis
    data1['Risk_Stratification'] = risk_stratification

    kmf_high_risk = KaplanMeierFitter()
    kmf_low_risk = KaplanMeierFitter()

# Fit data
    high_risk_data = data1[data1['Risk_Stratification'] == 'High Risk']
    low_risk_data = data1[data1['Risk_Stratification'] == 'Low Risk']

    kmf_high_risk.fit(high_risk_data['time_to_event'], event_observed=high_risk_data['event_occurred'], label='High Risk')
    kmf_low_risk.fit(low_risk_data['time_to_event'], event_observed=low_risk_data['event_occurred'], label='Low Risk')

# Plot survival curves
    kmf_high_risk.plot_survival_function()
    kmf_low_risk.plot_survival_function()

    plt.title('Survival Curves Stratified by Risk')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.savefig("survival_curve.png")


    val_c_index = concordance_index(y_test['time_to_event'], -predictions.flatten(), y_test['event_occurred'])

    return val_c_index  # Objective: maximize the concordance index

# Optuna study
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best hyperparameters
best_params = study.best_params
print('Best parameters:', best_params)
best_value = study.best_value
print('Best value (val_c_index):', best_value)





# Plot the optimization history
fig = ov.plot_optimization_history(study)

# Save the figure as an HTML file
pyo.plot(fig, filename='optimization_history_min.html', auto_open=False)


fig = ov.plot_parallel_coordinate(study)

pyo.plot(fig, filename='parallel_coordinate_min.html', auto_open=False)



best_trial = study.best_trial

