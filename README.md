# BirdsOnBikes

Dieses Repository beinhaltet die BirdsOnBike-homepage und unser TensorFlow Script

server.js liefert die Homepage aus und stellt einen API Endpoint für den Upload von Bilddateien bereit

Im 'client'-Ordner befinden sich die Dateien der Homepage

Hochgeladene Bilder werden sich zunächst im 'uploads'-Ordner gespeichert

Das TensorFlow-Script analysiert die Bilder im 'upload'-Ornder, findet die Künstliche Intelligenz einen oder mehrere Vögel auf dem Bild, wird es in den 'images'-Ordner verschoben. Sind keine Vögel auf dem Bild, wird es gelöscht