# SPT – eine Warteschlange, die sich selbst optimal sortiert – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-spt-scheduling-demo.streamlit.app/)**

Erstes Stück (Wurzel) der neuen **Klassische-Scheduling-Theorie-Linie** der "Konzepte"-Reihe für die Website
"Sebastian Hanisch – Operations Research und Machine Learning": $n$ Aufträge mit Bearbeitungszeit $p_j$ auf
**einer** Maschine, Ziel ist die Summe der Fertigstellungszeiten $\sum C_j$ zu minimieren ($1||\sum C_j$ in der
α|β|γ-Notation der Scheduling-Literatur).

**Einordnung in die Linie:** SPT (Shortest Processing Time first, Smith 1956) ist der einfachste Fall der ganzen
Linie – kein Suchverfahren, keine Heuristik, sondern eine **bewiesen optimale** Regel: einfach aufsteigend nach
Bearbeitungszeit sortieren. Jedes Folgestück hebt genau EINE Annahme dieses Beweises auf: Fristen statt keiner
Fristen (EDD, Moore-Hodgson), Gewichte statt Gleichgewichtung (WSPT), mehrere statt einer Maschine (Johnson-Regel,
LPT, Job Shop). Diese Demo führt zusätzlich **zwei Vehikel** ein, die die ganze Linie teilt: **Neutral**
(Aufträge mit Bearbeitungszeit) und **Werkstatt/Logistik** (dieselben Aufträge, aber in Familien mit Rüstzeit
beim Wechsel – dieselbe Idee wie die ATCS-Regel in `warehouse-transfer-demo`).
```
SPT (Wurzel: 1||ΣCⱼ, beweisbar optimal)                                          [dieses Stück]
 ├─ EDD (1||Lmax, dasselbe Beweismuster, andere Zielfunktion)                     [Folgestück]
 ├─ Moore-Hodgson (1||ΣUⱼ, gierig + schlechtesten Verspäteten entfernen)          [Folgestück]
 ├─ WSPT / Smith's Rule (1||ΣwⱼCⱼ, verallgemeinert SPT mit Gewichten)             [Folgestück]
 ├─ Johnson-Regel (F2||Cmax, zweite Maschine)                                     [Folgestück]
 ├─ LPT (Pm||Cmax, parallele Maschinen)                                           [Folgestück]
 └─ Job Shop (Konvergenzpunkt: Reihenfolge UND Maschinenwahl)                     [Folgestück]
```

Ergebnis in Kürze: **SPT trifft auf jeder getesteten Instanz (n = 2 bis 9) exakt das Minimum der Vollaufzählung –
der Beweis stimmt, nicht nur in der Theorie.** Bei 20 Aufträgen liegt SPT im Mittel **97.6 %** unter der
längste-zuerst-Reihenfolge (LPT) und **49.7 %** unter einer zufälligen Reihenfolge. Die Vollaufzählung selbst
wird schnell unpraktikabel: bei 9 Aufträgen braucht sie bereits über eine Sekunde, SPT bleibt bei rund 0,005 ms –
$n!$ wächst schneller als jede Potenz von $n$, $n \log n$ ist dagegen praktisch flach.
**Der ehrliche Bruch:** sobald Rüstzeiten zwischen Auftragsfamilien dazukommen (Vehikel Werkstatt/Logistik), setzt
der Beweis nicht mehr – SPT (das die Rüstzeiten weiterhin ignoriert) liegt messbar über dem echten Optimum, und
der Abstand wächst mit der Rüstzeit (0 % bei 0 Minuten, **28.9 %** bei 60 Minuten je Familienwechsel).

| Frage | Ergebnis (Mittel über 5 feste Instanzen, Seeds 100000–100004, mit je 3 Ketten-Seeds) |
|---|---|
| Standardfall (20 Aufträge) | ✅ SPT liegt **97.6 %** unter LPT und **49.7 %** unter einer zufälligen Reihenfolge |
| **Beweis gegen Vollaufzählung** | ✅ **100 %** Trefferquote bei n = 2 bis 9 – kein einziger Fall, in dem SPT nicht das Minimum trifft |
| **Rechenzeit** | ➖ Vollaufzählung bei n = 9 bereits über 1000 ms, SPT bei 0,005 ms – über 200.000-mal schneller |
| **Vehikel Werkstatt/Logistik** | ❌ Rüstzeit 0/5/15/30/60 Minuten: SPT liegt **0/1.6/6.6/14.3/28.9 %** über dem echten Optimum – der Beweis setzt keine Rüstzeiten voraus |

## Was die Demo zeigt

1. **SPT in Aktion** (Schritt-Slider): **Aufträge** (unsortierte Bearbeitungszeiten) → **Einplanen** (Regler
   "eingeplante Aufträge", Gantt-artiges Balkendiagramm baut sich auf) → **Ergebnis** (Σ-Fertigstellungskurve SPT
   gegen LPT).
2. **Was die Sortierung bringt:** SPT, LPT, zufällige Reihenfolge, Vollaufzählungs-Gegenprobe (n ≤ 9); auf dem
   Werkstatt/Logistik-Vehikel zusätzlich SPT mit Rüstzeiten gegen das echte Optimum mit Rüstzeiten.
3. **📐 Sweep** über Aufträge, Fristen-Anteil und -Streuung (für Folgestücke vorbereitet, hier nur zur
   Instanzkontrolle).
4. **🔬 Experimente auf Abruf:** Vollaufzählung gegen SPT über n = 2 bis 9 (Beweis-Check); Rechenzeit $n!$ gegen
   $n \log n$; Rüstzeit-Härtetest auf dem Werkstatt/Logistik-Vehikel.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (keine Rüstzeiten, gleiche Gewichte, keine
   Fristen, nur eine Maschine) mit Verweisen auf die Folgestücke.

Regler: Aufträge (2–60), **Vehikel** (Neutral/Werkstatt-Logistik – bei Werkstatt zusätzlich Rüstzeit und Anzahl
Familien), Seed der Instanz (+ 🎲), Seed der Kette (+ 🎲, steuert nur die zufällige Vergleichsreihenfolge – SPT
selbst ist deterministisch).

## Die zwei Vehikel (gelten für die ganze Linie)

- **Neutral** (`spt_scenario.py`): $n$ Aufträge mit Bearbeitungszeit $p_j \sim U(1, 100)$; Fälligkeit $d_j$ nach
  dem literaturüblichen TF/RDD-Schema (Tardiness-Faktor, Range der Fälligkeiten) und Gewicht $w_j \in \{1,2,4\}$
  sind für Folgestücke vorbereitet, in diesem Stück unbenutzt.
- **Werkstatt/Logistik** (`spt_scenario_logistik.py`): dieselben Bearbeitungszeiten/Fälligkeiten/Gewichte, aber
  jeder Auftrag gehört zu einer Familie; ein Familienwechsel kostet eine feste Rüstzeit (dieselbe Idee wie die
  ATCS-Regel in `warehouse-transfer-demo`). Rüstzeit 0 kollabiert exakt zum neutralen Vehikel (per Test belegt).
- Diese Demo etabliert bewusst **zwei echte, unabhängige Generatoren** statt eines einzigen mit Sonderfall (das
  sonst übliche Muster dieser Website, z. B. Kapazität → TSP-Spezialfall in `vrp-nachbarschaften-demo`) – die
  Frage "bleibt ein beweisbar optimales Verfahren optimal, sobald eine Annahme verletzt wird?" braucht eine
  Instanz, die diese Annahme wirklich verletzt, nicht nur einen Regler, der sie abschaltet.

## Modell und Verfahren

- **Instanz** (`spt_scenario.py`, `spt_scenario_logistik.py`): Bearbeitungszeiten, Fälligkeiten, Gewichte,
  Familien und Rüstzeit-Matrix, Seed-erzeugt wie jede andere Konzepte-Linie dieser Website.
- **SPT** (`spt_algorithm.py`): aufsteigend nach Bearbeitungszeit sortieren, $O(n \log n)$. Dazu die
  Brute-Force-Vollaufzählung (nur für kleine $n$) als unabhängige Gegenprobe, und die Rüstzeit-Variante für das
  Werkstatt/Logistik-Vehikel.
- **Auswertung** (`spt_evaluation.py`): Kennzahlen, Sweeps, Optimalitäts- und Timing-Messreihe,
  Rüstzeit-Härtetest.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Vermutung: "SPT bleibt eine gute Heuristik, auch wenn der Beweis nicht mehr gilt"** – auf dem
  Werkstatt/Logistik-Vehikel **teilweise bestätigt, aber nicht garantiert**: bei kleiner Rüstzeit ist der Abstand
  zum echten Optimum klein (1.6 % bei 5 Minuten), wächst aber deutlich mit der Rüstzeit (28.9 % bei 60 Minuten) –
  und einzelne Instanzen liegen noch höher (bis über 50 % im Einzelfall bei hoher Rüstlast, siehe Test). SPT
  bleibt also kein sicherer Ersatz für ein Verfahren, das Rüstzeiten selbst mit optimiert.
- **Die Vollaufzählung ist die einzige echte Gegenprobe**, aber praktisch nur bis $n \approx 9$ nutzbar – ab dort
  vertraut die Demo dem Beweis (Vertauschungsargument) statt einer weiteren Vollaufzählung. Das ist Absicht:
  jede andere Konzepte-Linie dieser Website macht denselben Wechsel von "gegen Brute-Force geprüft" zu
  "dem Beweis vertraut" ab einer bestimmten Größe.
- **Synthetische Instanzen:** Bearbeitungszeiten gleichverteilt, Fälligkeiten nach dem TF/RDD-Schema, keine
  Präzedenzen, ein Auftrag = eine Operation. Zeiten hängen vom Rechner ab, nur die Größenordnung zählt.

## Verifikation

- **Beweis gegen unabhängige Vollaufzählung:** für jede getestete Instanzgröße (n = 2 bis 9) und jede der 5
  festen Instanzen trifft SPT exakt das Minimum aller $n!$ Reihenfolgen – 100 % Trefferquote, sonst wäre entweder
  der Beweis oder die Implementierung falsch.
- **Rüstzeit-Variante gegen unabhängige Vollaufzählung** (mit Rüstzeiten statt ohne) und **Konsistenz-Test**:
  Rüstzeit 0 liefert exakt dieselbe Zielfunktion wie das neutrale Vehikel.
- **Handrechnung:** eine kleine, von Hand nachgerechnete Instanz (3 Aufträge) bestätigt SPT-Reihenfolge und
  Zielfunktionswert exakt.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Standardfall, Optimalitäts-Trefferquote, Rechenzeit,
  Rüstzeit-Härtetest, jeweils Mittel über die festen Sweep-Instanzen × Ketten; positive **und** negative
  Aussagen; Rechenzeiten nur als Größenordnung); alle 5 Presets geprüft; AppTest-Rauchtests (Voreinstellung,
  jedes Preset, jeder Schritt auf beiden Vehikeln, Würfel-Knöpfe, Permalink-Grenzen inkl. ungültigem Vehikel,
  Extremwerte, Experimente auf Abruf, Footer).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Experimente, 🚧 Grenzen, Mathe |
| `spt_algorithm.py` | SPT, Brute-Force-Gegenprobe, Rüstzeit-Variante |
| `spt_scenario.py` | Vehikel Neutral |
| `spt_scenario_logistik.py` | Vehikel Werkstatt/Logistik (Familien, Rüstzeit-Matrix) |
| `spt_constants.py` | Konstanten, Presets |
| `spt_evaluation.py` | Kennzahlen, Sweeps, Optimalitäts- und Timing-Messreihe, Rüstzeit-Härtetest |
| `spt_presets.py`, `spt_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | Beweis gegen Vollaufzählung, Szenario und Auswertung, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
