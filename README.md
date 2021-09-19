# BirdsOnBikes

Dieses Repository beinhaltet die BirdsOnBike-homepage und unser TensorFlow Script

server.js liefert die Homepage aus und stellt einen API Endpoint für den Upload von Bilddateien bereit

Im 'client'-Ordner befinden sich die Dateien der Homepage

Hochgeladene Bilder werden sich zunächst im 'uploads'-Ordner gespeichert

Das TensorFlow-Script analysiert die Bilder im 'upload'-Ornder, findet die Künstliche Intelligenz einen oder mehrere Vögel auf dem Bild, wird es in den 'images'-Ordner verschoben. Sind keine Vögel auf dem Bild, wird es gelöscht

## Dependencies

* Node [Quelle](https://nodejs.org/en/download/package-manager/)
* im Verzeichner 'server' ```npm install``` ausführen und dann den Serveer mit ```node server.js``` starten, der Server läuft dann auf localhost:3000
* TensorFlow
* im Verzeichnis 'TensorFlow' diese [Installationsanweisung](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) befolgen (Hinweis: am Besten den Installations-Weg mit Anaconda wählen, der ergab bei uns am wenigsten Probleme)
* mit ```python3 TensorFlowBirds.py``` das Programm ausführen, dass die Bidler analysiert