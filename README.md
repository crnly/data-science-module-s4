# Data Science — Semester 4

Aufarbeitung des Semesterprojekts SS 2026 an der HdM Stuttgart. Ziel ist die Klassifikation,
ob eine Person das aktuelle Modell einer Sport-/Triathlon-Uhr nutzt.

- [Notebook](data_science.ipynb): vollständige Analyse in sechs CRISP-DM-Kapiteln, mit kurzen
  Markdown-Texten und allen Ergebnissen.
- [CRISP-DM-Prozessübersicht](CRISP-DM-Prozessübersicht.md): ausführliche Begründungen,
  Prozessdiagramm und technischer Datenfluss.

## Vorgehen

Originaldaten prüfen → Training und Validation nach IDs trennen → logistische Ausgangsbaseline
und vier Aufbereitungsvarianten vergleichen → drei Modelle, Feature Selection und begrenztes
Tuning prüfen → festes Modell auf der gelieferten Testdatei bewerten.

## Ergebnis

Gewählt wurde **Logistische Regression mit Feature Selection** und der Aufbereitung **Bereinigt**.
Die Auswahl gilt für den festgelegten Validierungssplit. Eine zusätzliche Sensitivitätsprüfung
zeigt, wie sich die Rangfolge bei anderen Aufteilungen verändert.

| Gelieferter Kurs-Test · 500 Zeilen | Wert |
|---|---:|
| Accuracy | 0.790 |
| Macro-F1 | 0.780 |

465 Kennungen kommen in beiden Originaldateien vor; ihre Bedeutung über die Dateien hinweg
ist ungeklärt. Die Ergebnisse sind daher kein uneingeschränkter unabhängiger Leistungsnachweis
und keine Prognose zukünftiger Käufe.

## Lokal ausführen

Im Terminal in den Projektordner wechseln und die Umgebung anlegen; geprüft mit Python 3.14.4:

```bash
cd data-science-module-s4
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`data_science.ipynb` in Jupyter oder VS Code öffnen, den Python-Interpreter aus `.venv` als
Kernel auswählen und „Restart Kernel and Run All“ verwenden. Das Arbeitsverzeichnis muss
die Repositorywurzel sein.
Falls noch kein Notebook-Editor vorhanden ist: `python -m pip install jupyterlab`, danach
`jupyter lab`. Die beiden Original-CSV-Dateien liegen unverändert in `data/`.

## Einordnung

Die Analyse klassifiziert den beobachteten Nutzungsstatus anhand der vorliegenden Daten.
Code und Ergebnisse sind vollständig im Notebook enthalten; die Prozessübersicht erläutert
die methodischen Entscheidungen. Eine Lizenz wurde noch nicht gewählt.
