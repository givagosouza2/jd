import streamlit as st
import pandas as pd
import numpy as np
from scipy import signal
import io
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')
st.title("Análise de Aceleração 3D")
c1, c2 = st.columns(2)
with c1:
    # Upload do arquivo
    uploaded_file = st.file_uploader(
        "Faça upload de um arquivo .txt com cabeçalho e 4 colunas (tempo em ms, acc_x, acc_y, acc_z)", type=["txt", "csv"]
    )

    if uploaded_file is not None:
        # Leitura do arquivo e ajuste de separador
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        df.columns = df.columns.str.strip()

        # Verifica se tem ao menos 4 colunas
        if df.shape[1] >= 4:
            # Converte tempo para segundos (de ms para s)
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

            c1, c2, c3 = st.columns(3)
            with c1:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(t_interp, detrended_x, color='black')
                ax.set_xlabel("Tempo (s)")
                ax.set_ylabel("Aceleração")
                st.pyplot(fig)
            with c2:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(t_interp, detrended_y, color='black')
                ax.set_xlabel("Tempo (s)")
                ax.set_ylabel("Aceleração")
                st.pyplot(fig)
            with c3:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(t_interp, detrended_z, color='black')
                ax.set_xlabel("Tempo (s)")
                ax.set_ylabel("Aceleração")
                st.pyplot(fig)
