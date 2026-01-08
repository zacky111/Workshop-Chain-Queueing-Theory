import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SUM import SummationMethod

st.set_page_config(layout="wide", page_title="SUM — interfejs")

# inicjalizacja w session_state
if "sm" not in st.session_state:
    st.session_state.sm = SummationMethod()

sm = st.session_state.sm

# Pomocnicze funkcje parsujące
def parse_matrix(text, rows=None, cols=None, dtype=float):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    mat = []
    for ln in lines:
        row = [dtype(x.strip()) for x in ln.split(",") if x.strip() != ""]
        mat.append(row)
    arr = np.array(mat, dtype=dtype)
    if rows is not None and arr.shape[0] != rows:
        raise ValueError(f"Niepoprawna liczba wierszy: oczekiwano {rows}, otrzymano {arr.shape[0]}")
    if cols is not None and arr.shape[1] != cols:
        raise ValueError(f"Niepoprawna liczba kolumn: oczekiwano {cols}, otrzymano {arr.shape[1]}")
    return arr

st.title("SUM — interfejs")

# precision control
precision = st.sidebar.slider("Miejsca po przecinku", 0, 6, 3)
st.sidebar.write("Uruchom obliczenia w zakładce Wyniki")

tab1, tab2, tab3 = st.tabs(["Ustawienia systemu", "Przegląd parametrów", "Wyniki"])

with tab1:
    st.header("Ustawienia systemu")
    col1, col2 = st.columns(2)
    with col1:
        r = st.number_input("Liczba klas r", min_value=1, value=int(sm.r), step=1)
        n = st.number_input("Liczba węzłów n", min_value=1, value=int(sm.n), step=1)
        K_text = st.text_input("K (lista oddzielona przecinkami)", value=",".join(map(str, sm.K.tolist())))
        service_text = st.text_input("service_type (lista)", value=",".join(map(str, sm.service_type.tolist())))
        m_text = st.text_input("m (lista kanałów)", value=",".join(map(str, sm.m.tolist())))
    with col2:
        st.write("mi (macierz n x r). Wklej wiersze oddzielone nową linią, wartości przecinkami.")
        mi_default = "\n".join([",".join(map(str, row)) for row in sm.mi.tolist()])
        mi_text = st.text_area("mi", value=mi_default, height=150)
        st.write("p — macierze przejścia. Wpisz dla każdej klasy r osobną macierz (n wierszy), oddziel macierze pustą linią.")
        p_default = "\n\n".join(["\n".join([",".join(map(str, row)) for row in sm.p[idx]]) for idx in range(sm.r)])
        p_text = st.text_area("p", value=p_default, height=250)

    if st.button("Zastosuj zmiany"):
        try:
            params = {}
            params["r"] = int(r)
            params["n"] = int(n)
            params["K"] = [float(x.strip()) for x in K_text.split(",") if x.strip() != ""]
            params["service_type"] = [int(x.strip()) for x in service_text.split(",") if x.strip() != ""]
            params["m"] = [int(x.strip()) for x in m_text.split(",") if x.strip() != ""]
            params["mi"] = parse_matrix(mi_text, rows=int(n), cols=int(r)).tolist()

            # parse p: split blocks by empty line
            p_blocks = [b for b in p_text.split("\n\n") if b.strip()]
            if len(p_blocks) != int(r):
                st.warning(f"Oczekiwano {r} macierzy p (po jednej na klasę), znaleziono {len(p_blocks)}.")
            p_list = []
            for block in p_blocks:
                mat = parse_matrix(block, rows=int(n), cols=int(r)).tolist()
                p_list.append(mat)
            params["p"] = p_list

            # ustaw parametry
            sm.set_params(params)
            st.success("Parametry zastosowane i przeliczono e.")
        except Exception as exc:
            st.error(f"Błąd podczas parsowania parametrów: {exc}")

with tab2:
    st.header("Przegląd ustawionych parametrów")
    st.subheader("Ogólne")
    pd_params = pd.DataFrame({
        "r": [int(sm.r)],
        "n": [int(sm.n)],
        "epsilon": [sm.epsilon],
        "num_of_iterations": [sm.num_of_iterations]
    })
    st.dataframe(pd_params.round(precision))
    st.subheader("mi")
    st.dataframe(pd.DataFrame(sm.mi).round(precision))
    st.subheader("K")
    st.write(pd.Series(sm.K).round(precision))
    st.subheader("service_type")
    st.write(pd.Series(sm.service_type))
    st.subheader("m")
    st.write(pd.Series(sm.m))

with tab3:
    st.header("Wyniki")
    if st.button("Uruchom obliczenia"):
        sm.run_iteration_method_for_Lambda_r()
        sm.calculate_K_ir()
        sm.calculate_T_ir()
        st.success("Obliczenia zakończone")

    st.subheader("e_ir (visit ratios)")
    st.dataframe(pd.DataFrame(sm.e, columns=[f"r{j+1}" for j in range(sm.r)]).round(precision))

    st.subheader("Lambdas")
    st.write(pd.Series(sm.lambdas, index=[f"r{j+1}" for j in range(sm.r)]).round(precision))

    st.subheader("K_ir")
    st.dataframe(pd.DataFrame(sm.K_ir, columns=[f"r{j+1}" for j in range(sm.r)]).round(precision))

    st.subheader("T_ir")
    st.dataframe(pd.DataFrame(sm.T_ir, columns=[f"r{j+1}" for j in range(sm.r)]).round(precision))

    fig, ax = plt.subplots(figsize=(6, 4))
    c = ax.imshow(sm.K_ir, cmap="viridis", aspect="auto")
    ax.set_title("K_ir heatmap")
    ax.set_xlabel("Klasa r")
    ax.set_ylabel("Węzeł i")
    plt.colorbar(c, ax=ax)
    st.pyplot(fig)