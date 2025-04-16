import streamlit as st
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')
st.title("Análise de Aceleração 3D")

# Upload do arquivo
uploaded_file = st.file_uploader(
    "Faça upload de um arquivo .txt com cabeçalho e 4 colunas (tempo em ms, acc_x, acc_y, acc_z)", type=["txt", "csv"]
)

# Funções de filtro
def butter_filter(data, cutoff, fs, btype, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype=btype)
    return signal.filtfilt(b, a, data)

if uploaded_file is not None:
    # Leitura do arquivo e ajuste de separador
    df = pd.read_csv(uploaded_file, sep=None, engine='python')
    df.columns = df.columns.str.strip()

    if df.shape[1] >= 4:
        tempo = df.iloc[:, 0].values / 1000.0
        acc_x = df.iloc[:, 1].values
        acc_y = df.iloc[:, 2].values
        acc_z = df.iloc[:, 3].values

        # Interpolação para 100 Hz
        fs = 100
        t_min, t_max = tempo[0], tempo[-1]
        t_interp = np.arange(t_min, t_max, 1/fs)

        interp_x = np.interp(t_interp, tempo, acc_x)
        interp_y = np.interp(t_interp, tempo, acc_y)
        interp_z = np.interp(t_interp, tempo, acc_z)

        # Detrend
        detrended_x = signal.detrend(interp_x)
        detrended_y = signal.detrend(interp_y)
        detrended_z = signal.detrend(interp_z)

        # Filtros
        low_x = butter_filter(detrended_x, 2, fs, 'low')
        high_x = butter_filter(detrended_x, 2, fs, 'high')

        low_y = butter_filter(detrended_y, 2, fs, 'low')
        high_y = butter_filter(detrended_y, 2, fs, 'high')

        low_z = butter_filter(detrended_z, 2, fs, 'low')
        high_z = butter_filter(detrended_z, 2, fs, 'high')

        for eixo, original, low, high in zip(['X', 'Y', 'Z'],
                                             [detrended_x, detrended_y, detrended_z],
                                             [low_x, low_y, low_z],
                                             [high_x, high_y, high_z]):
            st.subheader(f"Eixo {eixo}")
            fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
            axs[0].plot(t_interp, original, color='black')
            axs[0].set_ylabel("Original")
