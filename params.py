# params.py
# Parametry sieci i klasy klientów.
# Bazowane na: "Propozycja tematu - sieci kolejkowe.docx". :contentReference[oaicite:1]{index=1}

# Liczba klientów (zamknięta populacja) przypisana do każdej klasy:
POPULATION = {
    1: 5,   # klasa 1 — uszkodzenia elektryczne
    2: 4,   # klasa 2 — uszkodzenia mechaniczne
    3: 3,   # klasa 3 — mieszane
    4: 2    # klasa 4 — uproszczone zlecenia
}

# Węzły (numery 1..8) i ich typy:
# typ 1 -> FIFO single-server (kolejka, 1 serwer)
# typ 3 -> IS infinite-server (brak kolejek, każdy od razu zaczyna obsługę)
NODE_TYPES = {
    1: 1,  # Przyjmowanie zgłoszenia (FIFO)
    2: 1,  # Dział elektryczny (FIFO)
    3: 1,  # Dział mechaniczny (FIFO)
    4: 3,  # Testy elektryczne - automatyczne (IS)
    5: 3,  # Testy mechaniczne - automatyczne (IS)
    6: 1,  # Wycena/dokumentacja (FIFO)
    7: 3,  # Obsługa klienta - wydanie urządzenia (IS)
    8: 3   # Stała eksploatacja (IS)
}

# Domyślne średnie szybkości obsługi (lambda = 1/mean_service_time).
# Dla rozkładu wykładniczego: time ~ Exp(rate)
SERVICE_RATES = {
    1: 1/3.0,   # node 1: średnio 3.0 czasu na obsługę
    2: 1/20.0,  # node 2: np. naprawa elektryczna średnio 20.0
    3: 1/25.0,  # node 3: naprawa mechaniczna średnio 25.0
    4: 1/5.0,   # node 4: testy elektryczne automatyczne średnio 5.0
    5: 1/6.0,   # node 5: testy mechaniczne automatyczne średnio 6.0
    6: 1/4.0,   # node 6: wycena/dokumentacja średnio 4.0
    7: 1/2.0,   # node 7: obsługa klienta średnio 2.0
    8: 1/10.0   # node 8: stała eksploatacja średnio 10.0
}

# Prawdopodobieństwo skoku z 6 -> 8 (bez przejścia przez 7).
# Pozostała część (1 - P_6_to_8) idzie 6 -> 7 -> 8 (jeśli dana klasa w ogóle używa 7).
P_6_TO_8 = 0.3

# Trasy klas (lista węzłów do odwiedzenia; po dojściu do ostatniego węzła wraca do 1).
# UWAGA: trasy opisane sekwencyjnie. Gdy węzeł 6 występuje, routing może rozgałęziać się
# zgodnie z P_6_TO_8 — to obsługujemy w kodzie symulacji.
ROUTES = {
    1: [1, 2, 4, 6, 7, 8],                # klasa 1 — elektryczne
    2: [1, 3, 5, 6, 7, 8],                # klasa 2 — mechaniczne
    3: [1, 2, 3, 4, 5, 6, 7, 8],          # klasa 3 — mieszane
    4: [1, 6, 7, 8]                       # klasa 4 — uproszczone
}

# Czas symulacji (jednostki czasu)
SIM_TIME = 20000.0

# Ziarno generatora losowego (dla powtarzalności)
SEED = 42
