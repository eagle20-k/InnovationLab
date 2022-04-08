# Code Anleitung - Deep Learning zur Einzelgebäudedetektion
Für die Skripte, die während des Praktikums an der Universität Tübingen erarbeitet wurden. Der Code ist verfügbar unter: https://github.com/eagle20-k/InnovationLab

Verfasst von Katrin Wernicke, am 08.04.2022.


## 1. Ordnerstruktur

|Ordner       | Unterordner|Unterordner|Beschreibung|
|--------------|----------|---------|-----------|
|0_Documents|||*Code Manual, Präsentation, Praktikumsreport*|
|1_ModelEngineering|log_dir||*Eventfiles für jedes trainierte Model (für das Tensorboard)*|
|              |models||*trainierte Modellem TensorFlow SavedModel Format*|
|1_Training Data |geojson   |         |*Original Gebäudefootprints (Polygone)*|
|              |MUL       |         |*WorldView-3 Multispektraldaten, 200x200m Kacheln*|
|              |MUL-PanSharpen|        | *WorldView-3 Multispektraldaten, pansharpened, 200x200m Kacheln*|
|              |PAN|         |*WorldView-3 Pan-Band, 200x200m Kacheln*|
|              |summaryData|         |
|              |RGB-PanSharpen|        | *Worldview-3 RGB-Daten, pansharpened, 200x200m Kacheln*|
|              |tiles|         |*vorprozessierte Satellitenbilder- und Gebäudemaskenkacheln, 200x200m Kacheln, Inputordner für Modeltraining*|
|              |AOI_2_VEGAS_processed|annotations|*rasterisierte Gebäudemaskenkacheln als JPG und PNG, Satellitenbildkacheln als TIF*|
|              ||geojson|*Gebäudefootprints (Polygone)*|
|              ||RGB-PanSharpen|*WorldView-3 RGB-Daten, pansharpened, 200x200m Kacheln*|
||utilities||*Github Repository von SpaceNet zur Vorprozessierung für Trainingsdaten*|
|2_Kigali        |Data|Raster|*Orginal Rasterbild, reskaliertes Rasterbild*|
|              |    |SHP |*Gebäudefootprints (Polygone)*|
|              |Predictions|2022-03-29-tiles_m5_subset_|*Rasterbildkacheln, Predicted Gebäudemaskenkacheln, Mosaik, Prediction mit 2015_Pleiades_Kigali_subset_.tif und Model 5*|
|              ||2022-03-29-tiles_m16_subset_|*Rasterbildkacheln, Predicted Gebäudemaskenkacheln, Mosaik, Prediction mit 2015_Pleiades_Kigali_subset_.tif und Model 16*|
|              ||2022-03-29-tiles_m17_subset_|*Rasterbildkacheln, Predicted Gebäudemaskenkacheln, Mosaik, Prediction mit 2015_Pleiades_Kigali_subset_.tif und Model 17*|
|3_Notebooks|||*Jupyter Notebooks, Environment File*|
|4_SpaceNetUnet|||*Original Github Repository von Alexey Bogatyrev*|


<div style="page-break-after: always;"></div>


## 2. Umgebungseinrichtung: 

### Miniconda

Für dieses Projekt wurde die Kommandozeile von Miniconda verwendet. 

>„Die Anaconda-Distribution ist eine Kollektion von Software für wissenschaftliche Zwecke. Sie enthält eine Python-Installation, eine R-Installation, sowie den Paketmanager Conda, der zur Installation von Anaconda-Paketen benutzt werden kann. Da die komplette Anaconda-Distribution mit allen Paketen sehr viel Speicherplatz verbraucht, gibt es außerdem die Variante Miniconda, die nur Python, Conda sowie ein paar grundlegende Pakete enthält. Beide Varianten sind vollständig kostenlos und Open-Source.“
https://cluster.uni-siegen.de/omni/application-software/miniconda/

Miniconda kann unter folgendem Link heruntergeladen werden: 
https://docs.conda.io/en/latest/miniconda.html

### Environment
**BEACHTE: Die nächsten Schritte sollen in der Anaconda Prompt (miniconda3) ausgeführt werden, nicht in der Windows Eingabeaufforderung!**

Die Environment-Datei **environment.yml** beinhaltet alle benötigten Pakete und Bibliotheken, um den Code für die Gebäudedetektion ausführen zu können. Der Umgebungsname lautet: dl_SpaceNetUnet.
Dabei geht man wie folgt vor (in der Kommandozeile): 

Umgebung dl_SpaceNetUnet auf eigenem Gerät erstellen:
```
conda env create -f environment.yml
```

Umgebung aktivieren:
```
conda activate dl_SpaceNetUnet
```

Überprüfe, ob die Umgebung korrekt installiert wurde:
```
conda env list
```

Weitführende Informationen zu Conda Umgebungen sind hier zu finden:
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands

## 3. SpaceNet Datendownload und Vorprozessierung

Zur Vollständigkeit befindet sich auch dieses Kapitel im Code Manual. Mit der Übergabe meiner Daten am Ende des Praktikums werden auch die  vorprozessierten Daten weitergegeben und müssen daher nicht neu heruntergeladen und prozessiert werden.

### Download
Die Datensätze von SpaceNet sind **Amazon Web Services (AWS) Public Datasets**. Sie beinhalten sehr hochaufgelöste Satellitenbilder (WorldView 3) sowie hochqualitative Labels für grundlegende Kartierungsmerkmale wie Gebäudegrundrisse oder Straßennetze. Datensätze gibt es für die Städte Las Vegas, Paris, Shanghai, Khartoum und Rio de Janeiro. 
Für diese Arbeit wurde der Datensatz Las Vegas verwendet. 

Für ein Download der SpaceNet Daten ist ein AWS Account erforderlich. Außerdem wird die AWS Command Line Interface (AWS CLI) benötigt, Informationen zum Download und zur Einrichtung finden sich hier: 
https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html


Nach Erstellung eines AWS Accounts gibt man in die Kommandozeile ein: 

```
aws configure
```

Folgende Daten werden danach abgefragt: 

```
AWS Access Key ID [None]: *enter your Key ID here*
AWS Secret Access Key [None]: *enter your Key here*
Default region name [None]: us-west-1
Default output format [None]: json
```

Auf der SpaceNet-Seite ist ersichtlich, dass die Daten im Vorhinein schon in Trainings- und Testdaten unterteilt wurden. 

**AOI 2 – Vegas – Building Extraction Training**

Um verarbeitete 200mx200m-Kacheln von AOI 2 (23 GB) mit den zugehörigen Gebäude Footprints für das Training herunterzuladen:

```
aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_2_Vegas.tar.gz . 
```

**AOI 2 – Vegas – Building Extraction Testing**

Um verarbeitete 200mx200m-Kacheln von AOI 2 (7.9 GB) mit den zugehörigen Gebäude Footprints für das Testing herunterzuladen:
```
aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/AOI_2_Vegas_test_public.tar.gz . 
```

Anmerkung: Für diese Arbeit wurde nur der Trainingsdatensatz heruntergeladen und manuell, im Skript, in Trainings-, Validierungs- und Testsamples geteilt. 

### Vorprozessierung

SpaceNet stellt einige Skripte zur Vorprozessierung der Daten zur Verfügung. 
Dafür muss zunächst das enstprechende GitHub-Repository geclont werden; alternativ wird der Ordner auch mit Übergeben.

```
git clone https://github.com/SpaceNetChallenge/utilities.git
```

Dann sollte man in der Kommandozeile zu dem geclonten Ordner steuern (über *cd*) und dann folgenden Code ausführen:
```
python python/createDataSpaceNet.py E:\W_Katrin\AOI_2_Vegas\AOI_2_Vegas_Train 
	--srcImageryDirectory RGB-PanSharpen
	--outputDirectory E:\W_Katrin\AOI_2_Vegas\AOI_2_Vegas_Train\AOI_2_Vegas_processed 
	--annotationType PASCALVOC2012 
	--convertTo8Bit 
	--outputFileType JPEG 
	--imgSizePix 650
```
*Anmerkung: Da die Kacheln vor dem Trainieren auf 256x265 Pixel große Kacheln herunterskaliert werden, könnte man hier auch einmal versuchen, die imgSizePix auf 256 zu setzen.*

Für die Ausführung der SpaceNetUnet-Skripte sollten alle Trainingsdaten, also Satellitenbilder wie Labels, im selben Ordner liegen, daher werden diese im folgenden Schritt in den Ordner *tiles* kopiert. Dafür muss zunächst erst wieder der Ort in der Ordnerstruktur angesteuert werden, an dem dieser neue Ordner angelegt werden soll.
```
mkdir tiles
cd E:\W_Katrin\AOI_2_Vegas\AOI_2_Vegas_Train\AOI_2_Vegas_processed\annotations
copy *jpg E:\W_Katrin\AOI_2_Vegas\AOI_2_Vegas_Train\tiles
copy *segcls.tif E:\W_Katrin\AOI_2_Vegas\AOI_2_Vegas_Train\tiles 
```


## 4. Jupyter verwenden

In diesem Absatz soll kurz beschrieben werden, wie ein Jupyter Notebook gestartet werden kann. 
Zunächst sollte die conda-Environment in der Kommandozeile aktiviert werden. 
Dann sollte der Ordner, in dem man arbeiten möchte, in der Kommandozeile angesteuert werden.
Nun lässt sich einfach ein Jupyter Notebook oder das Jupyter Lab starten.

Hier nochmal als Code: 

```
conda activate environment
cd ZIELORDNER
jupyter lab
```



## 5. Notebooks 


Das Projekt ist in 4 Notebooks unterteilt. Im Folgenden werden ihre Inhalte kurz beschrieben. 

### 01_Model_Engineering
*Autoren: Alexey Bogatyrev (Hauptautor), Katrin Wernicke (Kleine Veränderungen)*

In diesem Notebook wird das U-Net Model gebaut, konfiguriert und anhand der SpaceNet Trainingsdaten von Las Vegas trainiert. 
Die Modelarchitektur wurde mit Tensorflow und Keras konstruiert.
Außerdem kann die Model-Performance mithilfe einer eingebauten Tensorflow-Applikation beobachtet und evaluiert werden.


### 02_Tiling_Prediction_Mosaicing
*Autoren: Katrin Wernicke, Jonas Knecht*

Dieses Notebook dient zur Klassifikation/Segmentation von hochaufgelösten Satellitenbilder. 
Die Satellitenbilder werden für die Klassifikation vorbereitet indem die Pixelwerte auf den Wertebereich 0-255 reskaliert werden und zu 256x256 Pixel große Kacheln zugeschnitten werden. 
Die klassifizierten Kacheln werden in einigen Schritten wieder zu einem Mosaic zusammengefügt und ihre räumliche Information angehängt. 


### 00_Model_Eval_Vis
*Autoren: Katrin Wernicke, Code übernommen von Alexey Bogatyrev*

Dieses Notebook dient vor allem der Auswertung und Visualiserung der Model Performance, es ist also nicht essentiell für die Gebäudesegmentation. Es wurde speziell für die Praktikumspräsentation am 31.03.2022 an der Universität Tübingen geschrieben.

Dieses Notizbuch verwendet ein Sample des SpaceNet-Datensatzes, die dem Modell noch nicht gezeigt wurde, um die Leistung des Modells mit verschiedenen Trainingskonfigurationen zu testen. 
Die Genauigkeitsmetrik F1-score wird berechnet. Außerdem wird ein Beispiel für eine vorhergesagte Maske visualisiert. 

### 00_PreparingTrainingData
*Autoren: Jonas Knecht*

Mit diesem Notebook lassen sich Trainingsdaten, die nicht aus dem SpaceNet-Datensatz stammen, für das Modeltraining vorbereiten. 
Dabei werden die Daten zunächst in ein Integer 8bit-Format gebracht und in ein gewünschtes Dateiformat konvertiert. 
Die resultierenden Daten werden dann in 256x256 Pixel große Kacheln zerkleinert. Dabei besteht auch die Möglichkeit eine Stride-Größe zu definieren, um mehr Trainingsdaten generieren zu können. 


## 6. Sonstiges
### GPU verwenden
Das Model lässt sich auf der CPU trainieren, für ein deutlich schnelleres Training kann man es auch auf der GPU trainieren. Die zeitliche Einsparung kann von mehr als 5 Minuten pro Epoche auf wenige Sekunden verringert werden. Der einzige Nachteil ist, dass man bei Verwendung der GPU nur klein bis medium große Batchgrößen verwenden kann (erfahrungsgemäß maximal 32).
Eine Möglichkeit ist es, das Notebook in Google Colab laufen zu lassen und die dort integrierte TPU oder GPU zu verwenden. 

Auf dem lokalen Computer kann man auch die Environment so konfigurieren, dass das Notebook auf der lokalen GPU ausgeführt wird. Dazu wird das Paket tensorflow-gpu verwendet. Eine genauere Anleitung findet sich unter folgendem Link:
https://www.techentice.com/how-to-make-jupyter-notebook-to-run-on-gpu/

*Anmerkung für den Rechner bei Gebhard: Die GPU-Environment wurde hier schon eingerichtet und lautet gpu2. Das Vorgehen:*
- *Miniconda: conda activate gpu2*
- *Miniconda: jupyter lab*
- *Im Juypter Notebook oben rechts das Kernel gpu2 auswählen*

### Hilfreiche Weblinks:
https://machinelearningmastery.com/

https://github.com/robmarkcole/satellite-image-deep-learning

