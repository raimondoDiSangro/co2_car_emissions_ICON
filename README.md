# Car Co2 emissions ICON

### Descrizione
Il sistema, dotato di interfaccia grafica, permette l'inserimento e l'analisi di alcune caratteristiche di un'automobile (Cilindrata, Numero di Cilindri, Consumi),
per stimare il valore di emissioni della vettura in oggetto.
Inoltre, in base alle caratteristiche tecniche inserite, il sistema mostrerà all'utente una serie di vetture con caratteristiche simili presenti nel dataset a disposizione.

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



### Librerie Utilizzate
- ```numpy```, ```sklearn```, ```pandas```, ```seaborn```, ```yellowbrick```

### Struttura
- il progetto è strutturato in questa maniera, all'interno della directory ```source```:
  1. ```main.py``` consente l'avvio del programma e lancia l'interfaccia grafica con cui l'utente può inserire i dati;
  2. ```models.py``` ospita i modelli di classificazione per effettuare la predizione delle emissioni di co2 di una vettura con date caratteristiche;
  3. ```clustering.py``` ospita i modelli per effettuare il clustering del dataset e restituire le vetture simili;
- la directory ```data``` ospita il dataset utilizzato, in formato ```.csv```


### DISCLAIMER PER STUDENTI UNIBA (E NON SOLO...)
Questo progetto, tra l'altro per certi versi molto simile ad altri progetti realizzati da altri studenti per il corso di Ingegneria della Conoscenza, non è stato un grande successo in fase di valutazione finale.

Il mio personale consiglio per questo esame è quello di approcciarlo in due modi diversi:
1. Dedicarsi anima e corpo per realizzare un sistema che sia originale, sperando anche che sia gradito dal Professore, preparando la teoria in maniera pressoché perfetta (per quanto gli strumenti a disposizione per farlo non siano stati eccelsi, almeno nel periodo in cui ho frequentato io), con il rischio di non veder ripagato in fase di valutazione il lavoro monstre alle spalle;
2. Proporre un progetto simile a questo, senza pretese di assoluta originalità, del quale però dovrete almeno essere in grado di spiegare le scelte tecniche effettuate.

Francamente, la scelta che ho fatto personalmente, è stata una sorta via di mezzo tra le due, in quanto avrei voluto fare un buon esame, dato il mio interesse verso questa materia. Purtroppo però, credo che fra mancanze di tempo, problemi personali e onestamente un corso soporifero, seppure completo di argomenti, questo non è stato possibile.
Quindi ho dovuto mettere da parte una preparazione perfetta e una totale originalità di progetto.
Il risultato è stato il seguente: la mia peggiore valutazione della carriera universitaria, del quale non vado assolutamente fiero.

Tutto questo per dire, che potete prendere spunto, idee o quant'altro da questo progetto, così come ho fatto io con altri progetti simili, con la consapevolezza del fatto che i risultati finali potrebbero non essere eccelsi.

