import streamlit as st
import numpy as np
import pandas as pd
import json
import os
from SUM import SummationMethod
import plotly.graph_objects as go
import plotly.express as px

CONFIG_PATH = "config.json"

st.set_page_config(page_title="Summation Method", layout="wide")
st.title("Summation Method - Zamknięta Sieć Kolejkowa BCMP")

name_classes = [
    "Uszkodzenia elektryczne",
    "Uszkodzenia mechaniczne",
    "Uszkodzenia Mieszane",
    "Uproszczone zlecenia"
  ]
name_nodes = [
    "Przyjęcie zgłoszenia",
    "Dział elektryczny",
    "Dział mechaniczny",
    "Testy elektryczne - automatyczne",
    "Testy mechaniczne - automatyczne",
    "Wycena/dokumentacja",
    "Obsługa klienta - wydanie urządzenia",
    "Stała eksploatacja"
  ]

# Inicjalizacja sesji
if 'sm' not in st.session_state:
    st.session_state.sm = SummationMethod(CONFIG_PATH if os.path.exists(CONFIG_PATH) else None)
if 'results_calculated' not in st.session_state:
    st.session_state.results_calculated = False

# TABY
tab0, tab1, tab2, tab3 = st.tabs(["Model", "Parametryzacja", "Podgląd Parametrów", "Uruchomienie & Wyniki"])

# ============== TAB 0: MODEL ==============
with tab0:
    st.header("Model Systemu - Sieć kolejkowa obsługująca warsztat naprawczy")
    
    st.subheader("Opis Systemu")
    st.write("""
    System modeluje **zamkniętą sieć kolejkową BCMP** z wieloma klasami użytkowników.
    
    **Cechy systemu:**
    - Sieć zamknięta: Stała liczba zgłoszeń krążących w systemie
    - Wiele klas: Każda klasa może mieć inne parametry i trasy
    - Wiele węzłów: Każdy węzeł ma własne charakterystyki obsługi
    - Różne typy węzłów: FIFO (M/M/m) oraz IS (Infinite Server)
    
    **Metoda obliczeń:**
    - Wykorzystuje **Summation Method** do iteracyjnego wyznaczania intensywności przepływu (λ) dla każdej klasy
    - Iteracyjnie dąży do spełnienia warunku równowagi przepływu
    - Wyznacza średnie liczby zgłoszeń w węzłach (K_ir) i czasy przebywania (T_ir)
    """)

    st.subheader("Węzły Systemu i Klasy Zgłoszeń")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tabela z nazwami węzłów
        nodes_df = pd.DataFrame({
            "Węzeł": [f"Węzeł {i+1}" for i in range(len(name_nodes))],
            "Nazwa": name_nodes
        })
        st.dataframe(nodes_df, width='stretch', hide_index=True)
    
    with col2:
        # Tabela z nazwami klas
        classes_df = pd.DataFrame({
            "Klasa": [f"Klasa {i+1}" for i in range(len(name_classes))],
            "Nazwa": name_classes
        })
        st.dataframe(classes_df, width='stretch', hide_index=True)
    
    st.divider()
    
    st.subheader("Przebiegi Klas przez Sieć")
    
    # Pobranie listy zdjęć z katalogu assets
    assets_path = "assets"
    if os.path.exists(assets_path):
        image_files = sorted([f for f in os.listdir(assets_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])
        
        if image_files:
            # Wyświetlanie zdjęć w kolumnach
            st.write("**Trasy przepływu dla poszczególnych klas:**")
            
            cols = st.columns(2)
            for idx, img_file in enumerate(image_files):
                with cols[idx % 2]:
                    img_path = os.path.join(assets_path, img_file)
                    st.image(img_path, width='stretch', caption="Klasa " + str(idx+1) + " - " + name_classes[idx])
        else:
            st.warning("Brak plików obrazów w katalogu /assets")
    else:
        st.warning("Katalog /assets nie został znaleziony. Utwórz folder 'assets' i umieść tam zdjęcia.")
    
    st.divider()
    
    st.subheader("Parametry Kluczowe")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        **R** - liczba klas użytkowników
        
        **I** - liczba węzłów w sieci
        
        **K** - średnia liczba zgłoszeń klasy w systemie
        
        **m_i** - liczba kanałów obsługi w węźle i
        """)
    
    with col2:
        st.write("""
        **μ_ir** - intensywność obsługi klasy r w węźle i
        
        **e_ir** - średnia liczba wizyt klasy r w węźle i
        
        **λ_r** - intensywność przepływu klasy r
        
        **K_ir** - średnia liczba zgłoszeń klasy r w węźle i
        """)
    
    st.divider()
    


# ============== TAB 1: PARAMETRYZACJA ==============
with tab1:
    st.header("Ustawienia Parametrów")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Podstawowe")
        r = st.number_input("Liczba klas (R):", min_value=1, max_value=10, value=int(st.session_state.sm.r))
        n = st.number_input("Liczba węzłów (I):", min_value=1, max_value=20, value=int(st.session_state.sm.n))
        epsilon = st.number_input("Epsilon (dokładność):", min_value=1e-10, max_value=1e-2, value=float(st.session_state.sm.epsilon), format="%.2e")
        num_iterations = st.number_input("Liczba iteracji:", min_value=10, max_value=1000, value=int(st.session_state.sm.num_of_iterations))
    
    with col2:
        st.subheader("Średnia liczba zgłoszeń (K)")
        K_values = st.session_state.sm.K.copy()
        for i in range(int(st.session_state.sm.r)):
            K_values[i] = st.number_input(f"Klasa {i+1}:", min_value=0, max_value=100, value=int(st.session_state.sm.K[i]), step=1, key=f"K_{i}")
    
    st.subheader("Kanały obsługi (m_i)")
    cols_m = st.columns(int(st.session_state.sm.n))
    m_values = st.session_state.sm.m.copy()
    for i, col in enumerate(cols_m):
        with col:
            m_values[i] = st.number_input(f"Węzeł {i+1}", min_value=1, max_value=100, value=int(st.session_state.sm.m[i]), key=f"m_{i}")
    
    st.subheader("Typ węzła (1=FIFO, 3=IS)")
    cols_type = st.columns(int(st.session_state.sm.n))
    service_type_values = st.session_state.sm.service_type.copy()
    for i, col in enumerate(cols_type):
        with col:
            service_type_values[i] = st.selectbox(f"Węzeł {i+1}", options=[1, 3], index=0 if st.session_state.sm.service_type[i] == 1 else 1, key=f"service_{i}")
    
    st.subheader("Intensywność obsługi (μ_ir)")
    
    # Inicjalizacja DataFramu
    mi_edited = pd.DataFrame(
        st.session_state.sm.mi, 
        columns=[f"Klasa {i+1}" for i in range(int(st.session_state.sm.r))], 
        index=[f"Węzeł {i+1}" for i in range(int(st.session_state.sm.n))]
    )
    
    # Selectbox do wyboru węzła
    selected_node = st.selectbox("Wybierz węzeł do edycji:", 
                                  options=[f"Węzeł {i+1}" for i in range(int(st.session_state.sm.n))],
                                  key="node_select")
    node_idx = int(selected_node.split()[1]) - 1
    
    # Edycja dla wybranego węzła
    st.write(f"**Edycja intensywności obsługi dla {selected_node}**")
    cols = st.columns(int(st.session_state.sm.r))
    for class_idx, col in enumerate(cols):
        with col:
            mi_edited.iloc[node_idx, class_idx] = st.number_input(
                f"Klasa {class_idx+1}",
                min_value=0.1,
                max_value=100.0,
                value=float(st.session_state.sm.mi[node_idx, class_idx]),
                step=0.1,
                key=f"mi_{node_idx}_{class_idx}"
            )
    
    # Pełna tabela do podglądu
    st.write("**Pełna tabela intensywności obsługi**")
    st.dataframe(mi_edited, width='stretch', key="mi_preview")
    
    # Przycisk zapisania
    col_save, col_reset = st.columns(2)
    with col_save:
        if st.button("Zapisz konfigurację", width='stretch'):
            st.session_state.sm.r = int(r)
            st.session_state.sm.n = int(n)
            st.session_state.sm.K = np.array(K_values)
            st.session_state.sm.m = np.array(m_values)
            st.session_state.sm.service_type = np.array(service_type_values, dtype=int)
            st.session_state.sm.mi = mi_edited.values.astype(float)
            st.session_state.sm.epsilon = epsilon
            st.session_state.sm.num_of_iterations = num_iterations
            st.session_state.sm.calculate_E()
            st.session_state.sm.save_config(CONFIG_PATH)
            st.session_state.results_calculated = False
            st.success("Konfiguracja zapisana!")
    
    with col_reset:
        if st.button("Resetuj do domyślnych", width='stretch'):
            st.session_state.sm = SummationMethod()
            st.session_state.results_calculated = False
            st.rerun()

# ============== TAB 2: PODGLĄD PARAMETRÓW ==============
with tab2:
    st.header("Podgląd Aktualnych Parametrów")
    
    # WIERSZ 1
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Parametry globalne")
        st.write(f"**Liczba klas:** {st.session_state.sm.r}")
        st.write(f"**Liczba węzłów:** {st.session_state.sm.n}")
        st.write(f"**Epsilon:** {st.session_state.sm.epsilon:.2e}")
        st.write(f"**Max iteracji:** {st.session_state.sm.num_of_iterations}")
    
    with col2:
        st.subheader("Średnia liczba zgłoszeń (K)")
        K_df = pd.DataFrame(st.session_state.sm.K, index=[f"Klasa {i+1}" for i in range(int(st.session_state.sm.r))], columns=["K"])
        st.dataframe(K_df, width='stretch')
    
    with col3:
        st.subheader("Kanały obsługi (m_i)")
        m_df = pd.DataFrame(st.session_state.sm.m, index=[f"Węzeł {i+1}" for i in range(int(st.session_state.sm.n))], columns=["Kanały"])
        st.dataframe(m_df, width='stretch')
    
    # WIERSZ 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Typ węzła")
        type_names = {1: "FIFO", 3: "IS"}
        service_df = pd.DataFrame([type_names.get(int(x), str(x)) for x in st.session_state.sm.service_type], index=[f"Węzeł {i+1}" for i in range(int(st.session_state.sm.n))], columns=["Typ"])
        st.dataframe(service_df, width='stretch')
    
    with col2:
        st.subheader("Intensywność obsługi (μ_ir)")
        mi_df = pd.DataFrame(st.session_state.sm.mi, columns=[f"Klasa {i+1}" for i in range(int(st.session_state.sm.r))], index=[f"Węzeł {i+1}" for i in range(int(st.session_state.sm.n))])
        st.dataframe(mi_df, width='stretch')

# ============== TAB 3: URUCHOMIENIE & WYNIKI ==============
with tab3:
    st.header("Uruchomienie Metody")
    
    if st.button("Uruchom Summation Method", width='stretch', key="run_button"):
        with st.spinner("Obliczanie..."):
            st.session_state.sm.reset_lambdas()
            st.session_state.sm.run_SUM(alpha=0.3)  # pełne iteracje
            st.session_state.sm.calculate_K_ir()
            st.session_state.sm.calculate_T_ir()
            st.session_state.results_calculated = True
        st.success(f"Obliczenia zakończone!")
    
    if st.session_state.results_calculated:
        st.subheader("Wyniki")
        
        # Wyniki tabelaryczne
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Lambdas (intensywność przepływu klas)**")
            lambda_df = pd.DataFrame(st.session_state.sm.lambdas, index=[f"Klasa {i+1}" for i in range(int(st.session_state.sm.r))], columns=["λ"])
            st.dataframe(lambda_df, width='stretch')
        
        with col2:
            st.write("**Suma K_ir dla każdej klasy**")
            K_sum = np.sum(st.session_state.sm.K_ir, axis=0)
            K_sum_df = pd.DataFrame(K_sum, index=[f"Klasa {i+1}" for i in range(int(st.session_state.sm.r))], columns=["K"])
            st.dataframe(K_sum_df, width='stretch')

        with col3:
            st.write("**e_ir (średnia liczba wizyt)**")
            e_df = pd.DataFrame(st.session_state.sm.e.round(3), columns=[f"Klasa {i+1}" for i in range(int(st.session_state.sm.r))], index=[f"Węzeł {i+1}" for i in range(int(st.session_state.sm.n))])
            st.dataframe(e_df, width='stretch')

        col1, col2 = st.columns(2)
        
        with col1: 
            st.write("**K_ir (średnia liczba zgłoszeń w węźle)**")
            K_ir_df = pd.DataFrame(st.session_state.sm.K_ir.round(3), columns=[f"Klasa {i+1}" for i in range(int(st.session_state.sm.r))], index=[f"Węzeł {i+1}" for i in range(int(st.session_state.sm.n))])
            st.dataframe(K_ir_df, width='stretch')
        
        with col2:
            st.write("**T_ir (średni czas przebywania w węźle)**")
            T_ir_df = pd.DataFrame(st.session_state.sm.T_ir.round(3), columns=[f"Klasa {i+1}" for i in range(int(st.session_state.sm.r))], index=[f"Węzeł {i+1}" for i in range(int(st.session_state.sm.n))])
            st.dataframe(T_ir_df, width='stretch')
        
        # Wizualizacje
        col1, col2 = st.columns(2)
        
        with col1:
            fig_lambda = px.bar(x=[f"Klasa {i+1}" for i in range(len(st.session_state.sm.lambdas))], 
                               y=st.session_state.sm.lambdas,
                               title="Lambdas (Intensywność przepływu)",
                               labels={"x": "Klasa", "y": "λ"})
            st.plotly_chart(fig_lambda, use_container_width=True)
        
        with col2:
            fig_K = px.bar(x=[f"Węzeł {i+1}" for i in range(int(st.session_state.sm.n))],
                          y=np.sum(st.session_state.sm.K_ir, axis=1),
                          title="Suma K_ir po węzłach",
                          labels={"x": "Węzeł", "y": "K"})
            st.plotly_chart(fig_K, use_container_width=True)
        
    else:
        st.info("Wciśnij przycisk 'Uruchom' aby obliczyć wyniki")
