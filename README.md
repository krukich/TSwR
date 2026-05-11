# Teoria sterowania w robotyce

## Opis projektu
Celem projektu jest implementacja i analiza algorytmu adaptacyjnego inspirowanego podejściem RMA (Rapid Motor Adaptation) dla robota kroczącego w środowisku symulacyjnym Genesis. Zadaniem modelu jest generowanie stabilnego ruchu robota w zmieniających się warunkach środowiskowych, takich jak różne wartości tarcia, śliskość podłoża oraz nierówności terenu.

W ramach projektu planowane jest wytrenowanie dwóch komponentów: polityki sterującej ruchem robota oraz modułu adaptacyjnego, który na podstawie historii stanów i akcji będzie estymował ukryty stan środowiska.

---

## Cele
- Implementacja polityki sterującej ruchem robota  
- Implementacja modułu adaptacyjnego  
- Trening modelu w środowisku Genesis  
- Porównanie działania modelu adaptacyjnego na zbiorze danych testowych. 

---

## Wytrenowane modele

Wytrenowane modele nie są przechowywane bezpośrednio w repozytorium ze względu na rozmiar plików checkpointów. Wszystkie aktualne modele znajdują się w osobnym katalogu na Google Drive:

[Google Drive - wytrenowane modele](https://drive.google.com/drive/folders/1x3uDen8MqZw5EVCSddOEvBN7ynM8qbgJ?usp=drive_link)

W każdym katalogu konkretnego modelu, razem z plikami checkpointów, znajduje się plik `description.txt`, który zawiera krótki opis modelu, konfigurację treningu oraz zalecany sposób użycia.

---

## Architektura
Projekt składa się z dwóch głównych komponentów:

- **Base Policy (π)** – generuje akcje na podstawie aktualnego stanu robota  
- **Adaptation Module (φ)** – estymuje latent \(z_t\) na podstawie historii stanów i akcji  

---

## Bibliografia
- RMA: Rapid Motor Adaptation for Legged Robots  
  https://arxiv.org/pdf/2107.04034

---

## Zespół
- Artsiom Kruk 
- Daniil Kavalevich
