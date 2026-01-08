import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SUM import SummationMethod

st.title("SUM — interfejs")

precision = st.sidebar.slider("Miejsca po przecinku", 0, 6, 3)
num_iter = st.sidebar.number_input("Maks. iteracji", min_value=1, value=200)
K_input = st.sidebar.text_input("K (lista, oddzielona przecinkami)", value="2,4,2,3")

if st.sidebar.button("Uruchom"):
    K_list = [float(x.strip()) for x in K_input.split(",") if x.strip()]
    sm = SummationMethod()
    if len(K_list) == sm.r:
        sm.K = np.array(K_list)
    sm.num_of_iterations = int(num_iter)
    sm.run_iteration_method_for_Lambda_r()
    sm.calculate_K_ir()
    sm.calculate_T_ir()

    cols = [f"r{j+1}" for j in range(sm.r)]
    idx = [f"i{i+1}" for i in range(sm.n)]

    st.subheader("e_ir (visit ratios)")
    st.dataframe(pd.DataFrame(sm.e, columns=cols, index=idx).round(precision))

    st.subheader("Lambdas")
    st.write(pd.Series(sm.lambdas, index=cols).round(precision))

    st.subheader("K_ir")
    st.dataframe(pd.DataFrame(sm.K_ir, columns=cols, index=idx).round(precision))

    st.subheader("T_ir")
    st.dataframe(pd.DataFrame(sm.T_ir, columns=cols, index=idx).round(precision))

    fig, ax = plt.subplots()
    c = ax.imshow(sm.K_ir, cmap="viridis", aspect="auto")
    ax.set_title("K_ir heatmap")
    ax.set_xlabel("Klasa r")
    ax.set_ylabel("Węzeł i")
    plt.colorbar(c, ax=ax)
    st.pyplot(fig)