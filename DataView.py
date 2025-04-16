import streamlit as st 
import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')
st.title("Análise de Aceleração 3D com Detecção de Rigidez")

# Upload do arquivo
uploaded_file = st.file_uploader(
    "Faça upload de um arquivo .txt ou .csv com cabeçalho e 4 colunas (tempo em ms, acc_x, acc_y, acc_z)", type=["txt", "csv"]
)

# Filtro Butterworth
def butter_filter(data, cutoff, fs, btype, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype=btype)
    return signal.filtfilt(b, a, data)

# PSD plot
def plot_psd(data, fs, eixo):
    f, Pxx = signal.welch(data, fs=fs, nperseg=1024)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.semilogy(f, Pxx, color='green')
    ax.set_xlabel('Frequência (Hz)')
    ax.set_ylabel('Densidade espectral')
    ax.set_title(f'Densidade espectral de potência - Eixo {eixo}')
    ax.grid(True)
    return fig, f, Pxx

# Band power
def band_power(frequencies, power_spectrum, fmin, fmax):
    mask = (frequencies >= fmin) & (frequencies <= fmax)
    return np.trapz(power_spectrum[mask], frequencies[mask])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=None, engine='python')
    df.columns = df.columns.str.strip()

    if df.shape[1] >= 4:
        tempo = df.iloc[:, 0].values / 1000.0
        acc_x = df.iloc[:, 1].values
        acc_y = df.iloc[:, 2].values
        acc_z = df.iloc[:, 3].values

        # Interpolação
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

        

        # PSD + análise espectral eixo X
        low_lim = st.number_input('Selecione o limite inferior da janela de interesse',value=0)
        high_lim = st.number_input('Selecione o limite superior da janela de interesse',value=0)
        fig_psd, f, Pxx = plot_psd(detrended_x[low_lim:high_lim], fs, 'X')
        st.pyplot(fig_psd)

        power_movement = (10**5)*band_power(f, Pxx, 0.3, 1.5)
        power_rigidity = (10**5)*band_power(f, Pxx, 2.0, 6.0)
        rigidity_index = power_rigidity / power_movement if power_movement > 0 else np.nan

        # Gráficos temporais
        fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        axs[0].plot(t_interp, detrended_x, color='black')
        axs[0].plot([t_interp[low_lim],t_interp[low_lim]],[-4, 4],'--b')
        axs[0].plot([t_interp[high_lim],t_interp[high_lim]],[-4, 4],'--b')
        axs[0].set_ylabel("Original")
        axs[1].plot(t_interp, low_x, color='blue')        
        axs[1].set_ylabel("Passa-Baixa 2 Hz")        
        axs[2].plot(t_interp, high_x, color='red')
        axs[2].set_ylabel("Passa-Alta 2 Hz")
        axs[2].set_xlabel("Tempo (s)")
        st.pyplot(fig)

        # Resultados
        st.subheader("Análise Quantitativa - Eixo X")
        
        st.markdown(f"- **Potência 0.3–1.5 Hz (movimento voluntário)**: `{power_movement:.3f}`")
        st.markdown(f"- **Potência 2–6 Hz (rigidez roda dentada)**: `{power_rigidity:.3f}`")
        st.markdown(f"- **Índice de rigidez relativa**: `{rigidity_index:.3f}`")
