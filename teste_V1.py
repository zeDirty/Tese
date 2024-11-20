import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras import layers
from matplotlib import pyplot as plt
from tensorflow import keras

df_sem_anomalia_path = "C:\\Users\\José Sanches\\Desktop\\11NOV\\Impacto_forte_Corte\\semanomalia_80.csv"
df_com_anomalia_path = "C:\\Users\\José Sanches\\Desktop\\11NOV\\Impacto_forte_Corte\\comanomalia_20.csv"

# Carregar dados e definir 'datetime' como índice temporal
df_small_noise = pd.read_csv(df_sem_anomalia_path)
df_small_noise = df_small_noise.set_index('datetime')  # Definir 'datetime' como índice
df_small_noise.index = pd.to_datetime(df_small_noise.index)  # Converter para formato datetime

df_daily_jumpsup = pd.read_csv(df_com_anomalia_path)
df_daily_jumpsup = df_daily_jumpsup.set_index('datetime')  # Definir 'datetime' como índice
df_daily_jumpsup.index = pd.to_datetime(df_daily_jumpsup.index)  # Converter para formato datetime

# Remover a coluna redundante 'timestamp'
df_small_noise = df_small_noise.drop(columns=['timestamp'])
df_daily_jumpsup = df_daily_jumpsup.drop(columns=['timestamp'])

#print(df_small_noise.head())
#print(df_daily_jumpsup.head())


#fig, ax = plt.subplots()
#df_small_noise.plot(legend=False, ax=ax)
#plt.show()

#fig, ax = plt.subplots()
#df_daily_jumpsup.plot(legend=False, ax=ax)
#plt.show()


# Normalize and save the mean and std we get,
# for normalizing test data.
#training_mean = df_small_noise.mean()
#training_std = df_small_noise.std()

# Subamostrar por número de entradas
chunk_size = 5
df_small_noise_resampled = df_small_noise.groupby(np.arange(len(df_small_noise)) // chunk_size).mean()
df_daily_jumpsup_resampled = df_daily_jumpsup.groupby(np.arange(len(df_daily_jumpsup)) // chunk_size).mean()

# Normalize and save the mean and std we get,
# for normalizing test data (após a subamostragem)
training_mean = df_small_noise_resampled.mean()
training_std = df_small_noise_resampled.std()

# Normalizar os dados subamostrados
df_training_value_resampled = (df_small_noise_resampled - training_mean) / training_std
print("Number of training samples (after resample):", len(df_training_value_resampled))

# Normalizar os dados subamostrados (teste)
df_test_value_resampled = (df_daily_jumpsup_resampled - training_mean) / training_std
print("Number of test samples (after resample):", len(df_test_value_resampled))

TIME_STEPS = 100
#1500 1min com chunk_size=2
#1200 2min com chunk_size=5
#1500 5min com chunk_size=10

'''
# Generated training sequences for use in the model. ORIGINAL
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)
'''

# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(0, len(values) - time_steps + 1, time_steps):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

'''
# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS, overlap=100):
    step = time_steps - overlap  # Define o deslocamento entre janelas
    output = []
    for i in range(0, len(values) - time_steps + 1, step):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)
'''

x_train = create_sequences(df_training_value_resampled.values)
print("Training input shape: ", x_train.shape)


model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        layers.Conv1D(
            filters=32,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Conv1DTranspose(
            filters=16,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()



history = model.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ],
)


plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()



# Get train MAE loss.
x_train_pred = model.predict(x_train)
train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

plt.hist(train_mae_loss, bins=50)
plt.xlabel("Train MAE loss")
plt.ylabel("No of samples")
plt.show()

# Get reconstruction loss threshold.
threshold = np.max(train_mae_loss)
print("Reconstruction error threshold: ", threshold)


# Checking how the first sequence is learnt
plt.plot(x_train[0])
plt.plot(x_train_pred[0])
plt.show()


'''
df_test_value = (df_daily_jumpsup - training_mean) / training_std
fig, ax = plt.subplots()
df_test_value.plot(legend=False, ax=ax)
plt.show()
'''

# Create sequences from test values.
x_test = create_sequences(df_test_value_resampled.values)
print("Test input shape: ", x_test.shape)

# Get test MAE loss.
x_test_pred = model.predict(x_test)
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
test_mae_loss = test_mae_loss.reshape((-1))

plt.hist(test_mae_loss, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.show()

#multiplicar 1.2
# Detect all the samples which are anomalies.
anomalies = test_mae_loss > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))

'''
# data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
anomalous_data_indices = []
for data_idx in range(TIME_STEPS - 1, len(df_test_value_resampled) - TIME_STEPS + 1):
    if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
        anomalous_data_indices.append(data_idx)
##verificar
'''


'''
# Obter as localizações das anomalias identificadas
anomaly_location = np.where(anomalies)[0]  # Índices onde as anomalias foram detectadas
anomalous_data_indices = []

# Mapear cada índice de anomalia para os índices reais no DataFrame
for anomaly_index in anomaly_location:
    start_idx = anomaly_index * TIME_STEPS
    end_idx = start_idx + TIME_STEPS
    anomalous_data_indices.extend(range(start_idx, end_idx))  # Guardar os índices reais

# Garantir que não há duplicação de índices
anomalous_data_indices = sorted(set(anomalous_data_indices))

# Selecionar os dados anómalos do DataFrame original
df_subset = df_daily_jumpsup_resampled.iloc[anomalous_data_indices]
print("Índices no df_subset (anomalias):", df_subset.index)
print("Conteúdo do df_subset:\n", df_subset)

# Gráfico com anomalias destacadas
fig, ax = plt.subplots()

# Plot dos dados originais
df_daily_jumpsup_resampled.plot(legend=False, ax=ax, color="blue", label="Dados Originais")

# Destacar os pontos anómalos em vermelho
ax.scatter(
    df_subset.index,  # Índices das anomalias
    df_subset["IMU.AccZ"],  # Valores das anomalias
    color="red",
    zorder=5,
    label="Anomalias",
)

# Adicionar legenda para claridade
ax.legend()

plt.show()
'''

anomaly_location = np.where(anomalies)
anomalous_data_indices = []
#for anomaly_index in range (len(anomaly_location)):
#    print("Anomaly index: ", anomaly_index)
#    print("Anomalies : ", anomaly_location[anomaly_index])
    #anomalous_data_indices.append([anomaly_index: anomaly_index + TIME_STEPS])
    
for position, value in enumerate(anomaly_location[0]):
    print('position: ',position)
    print('value: ',value)
    vector = np.arange(value * TIME_STEPS , value * TIME_STEPS + TIME_STEPS)
    #print('vetores: ', vector)
    #anomalous_data_indices.append(vector)
    anomalous_data_indices.extend(vector)

anomalous_data_indices = sorted(set(anomalous_data_indices))

df_subset = df_daily_jumpsup_resampled.iloc[anomalous_data_indices]
fig, ax = plt.subplots()
df_daily_jumpsup_resampled.plot(legend=False, ax=ax)
df_subset.plot(legend=False, ax=ax, color="r", linestyle="", marker="o")
plt.show()
