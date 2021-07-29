# Car Co2 emissions ICON

### Descrizione


### Dataset
- I dati a disposizione contengono specifiche tecniche di una serie di automobili.
 - Il dataset è stato scaricato dal Repository pubblico Canadese sulle emissioni delle automobili: https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64#wb-auto-6
- Contiene le specifiche tecniche di automobili in commercio negli ultimi sette anni in Canada.



### Contenuto dataset 
  1. ```make: marca della vettura ```
  2. ```model: modello della vettura ```
  3. ```vehicle_class: classe del veicolo (es. SUV, SEDAN, etc.)```
  4. ```engine_size: cilindrata del motore del veicolo in litri (es. 2.0)```
  5. ```cylinders: numero di cilindri del motore```
  6. ```transmission: tipo di trasmissione del veicolo (automatico o manuale, con indicazione sul numero di marce)```
  7. ```fuel_type: tipologia di carburante del veicolo (es. Diesel, benzina, benzina premium, etc.)```
  8. ```fuel_consumption_city: il consumo della vettura in ambito cittadino, espresso in l/100km```
  9. ```fuel_consumption_hwy: il consumo della vettura in ambito autostradale, espresso in l/100km```
  10. ```fuel_consumption_comb(l/100km): il consumo della vettura in ambito misto, espresso in l/100km```
  11. ```fuel_consumption_comb(mpg): il consumo della vettura in ambito misto, espresso in miglia per gallone```
  12. ```co2_emissions: la quantità di co2 emessa dal veicolo in g/km (es. 120 g/km)```



### Requisiti
- librerie ```numpy```, ```sklearn```, ```pandas```, ```seaborn```

### Struttura
- il progetto è strutturato in questa maniera, all'interno della directory ```src```:
  1. ```main.py``` consente l'avvio del programma e lancia l'interfaccia grafica con cui l'utente può inserire i dati;
  2. ```classification_models.py``` ospita i modelli di classificazione per effettuare la predizione delle emissioni di co2 di una vettura con date caratteristiche;
- la directory ```data``` ospita il dataset utilizzato, in formato ```.csv```;