"""SPT (Shortest Processing Time first) - eine Warteschlange, die sich selbst optimal sortiert - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Wurzel der neuen Konzepte-Linie "Klassische Scheduling-Theorie": n Aufträge auf einer Maschine, Ziel ist die Summe
der Fertigstellungszeiten zu minimieren (1||ΣCⱼ in der α|β|γ-Notation der Scheduling-Literatur - α: eine Maschine,
β: keine Nebenbedingungen, γ: das Ziel). SPT (aufsteigend nach Bearbeitungszeit sortieren) ist dafür BEWEISBAR
optimal - ein Vertauschungsargument, kein Suchverfahren. Siehe README für die Einordnung in die Linie.

Lauffähig mit: streamlit run app.py
"""

from dataclasses import replace

import numpy as np
import streamlit as st

import spt_algorithm as A
import spt_constants as C
import spt_scenario_logistik as SL
from spt_evaluation import Settings, SWEEP_LABELS, analyse, instance, optimality_check, run_config, setup_gap, setup_gap_sweep, sweep, timing_sweep
from spt_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_chain_seed, randomize_seed, sync_query_params
from spt_visualization import build_completion_curve, build_schedule, build_setup_gap, build_sweep, build_timing

st.set_page_config(page_title="SPT-Scheduling – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _optimality():
    return optimality_check()


@st.cache_data(show_spinner=False)
def _timing():
    return timing_sweep()


@st.cache_data(show_spinner=False)
def _setup_gap_sweep(n, n_families):
    return setup_gap_sweep(n=n, n_families=n_families)


def _fmt_int(x):
    return f"{int(round(x)):,}".replace(",", ".")


st.title("⏱️ SPT – eine Warteschlange, die sich selbst optimal sortiert")
st.markdown(
    r"""
**n Aufträge auf einer Maschine, eine Reihenfolge gesucht, die die Summe der Fertigstellungszeiten minimiert**
($1||\sum C_j$ in der Notation der Scheduling-Theorie). Die Antwort ist **SPT** (Shortest Processing Time first):
einfach aufsteigend nach Bearbeitungszeit sortieren – kein Suchverfahren, keine Heuristik, sondern **beweisbar
optimal**. Der Beweis ist ein Vertauschungsargument: in jeder Reihenfolge, die nicht sortiert ist, gibt es zwei
benachbarte Aufträge $i$ vor $j$ mit $p_i > p_j$ – vertauscht man sie, ändert sich nur die Summe der beiden
Fertigstellungszeiten, und die wird dabei um genau $p_i - p_j > 0$ kleiner. Jede unsortierte Reihenfolge lässt sich
also verbessern; SPT ist der einzige Zustand, an dem keine Vertauschung mehr hilft.
"""
)
st.caption(
    "Wurzel der neuen Konzepte-Linie „Klassische Scheduling-Theorie“ - führt die α|β|γ-Notation ein, die alle "
    "Folgestücke verwenden (Earliest Due Date, Moore-Hodgson, gewichtete Regeln, Johnson-Regel, parallele Maschinen, "
    "Job Shop). Zwei Vehikel: **Neutral** (Aufträge mit Bearbeitungszeit) und **Werkstatt/Logistik** (dieselben "
    "Aufträge, aber in Familien mit Rüstzeit beim Wechsel) - der Umschalter ist in der Seitenleiste."
)

with st.expander("So funktioniert SPT", expanded=True):
    st.markdown(
        r"""
1. **Sortieren.** Alle Aufträge aufsteigend nach Bearbeitungszeit $p_j$ ordnen - fertig. $O(n \log n)$, kein Suchverfahren nötig.
2. **Warum das optimal ist.** Vertauschungsargument: zwei benachbarte Aufträge mit $p_i > p_j$ tauschen verringert die Summe der Fertigstellungszeiten um $p_i - p_j$ - jede nicht sortierte Reihenfolge lässt sich also verbessern.
3. **Was gemessen wird.** Der Abstand einer Reihenfolge zu SPT in Prozent; für kleine $n$ zusätzlich die Vollaufzählung aller $n!$ Reihenfolgen als unabhängige Gegenprobe.
4. **Die Grenze der Annahme.** SPT setzt voraus, dass die Bearbeitungszeiten unabhängig von der Reihenfolge sind - keine Rüstzeiten. Das Vehikel „Werkstatt/Logistik“ prüft, was passiert, wenn das nicht mehr stimmt.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
cols = st.columns(len(preset_names))
for col, name in zip(cols, preset_names):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_jobs = st.slider("Aufträge", *bounds("n_slider"), key="n_slider", step=C.N_STEP,
                        help=f"Anzahl der Aufträge. Bis {C.BRUTE_FORCE_MAX_N} läuft die Vollaufzählung aller n! Reihenfolgen live mit.")
    vehicle = st.radio("Vehikel", list(C.VEHICLE_LABELS), key="vehicle_radio", format_func=lambda k: C.VEHICLE_LABELS[k],
                        help="Neutral: nur Bearbeitungszeiten. Werkstatt/Logistik: dieselben Aufträge, zusätzlich in Familien mit Rüstzeit beim Wechsel.")
    if vehicle == "logistik":
        setup_time = st.slider("Rüstzeit je Familienwechsel (Minuten)", *bounds("setup_time_slider"), key="setup_time_slider",
                                help="0 Minuten kollabiert exakt zum neutralen Vehikel (siehe Test/Messreihe).")
        n_families = st.slider("Auftragsfamilien", *bounds("n_families_slider"), key="n_families_slider",
                                help="Weniger Familien bei gleicher Auftragszahl bedeutet mehr Wechsel und damit mehr Rüstzeit insgesamt.")
    else:
        setup_time, n_families = C.DEFAULT_SETUP_TIME, C.DEFAULT_N_FAMILIES
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Seed für die Bearbeitungszeiten.")
    chain_seed = st.number_input("Zufalls-Seed der Kette", *bounds("chain_seed_input"), key="chain_seed_input", step=1,
                                  help="Steuert nur die zufällige Vergleichs-Reihenfolge - SPT selbst ist deterministisch (kein Zufall im Kern).")
    st.button("🎲 Neue Kette würfeln", width="stretch", on_click=randomize_chain_seed, help="Würfelt einen neuen Seed für die Zufalls-Vergleichsreihenfolge.")

sync_query_params({"n_slider": int(n_jobs), "seed_input": int(seed), "chain_seed_input": int(chain_seed), "vehicle_radio": vehicle,
                    "setup_time_slider": int(setup_time), "n_families_slider": int(n_families)})

settings = Settings(int(n_jobs), int(seed), int(chain_seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
inst = a.inst
p = inst.p
data_key = (settings, vehicle, setup_time, n_families)

if vehicle == "logistik":
    linst = SL.generate(int(n_jobs), int(seed), n_families=int(n_families), setup_time=int(setup_time))
    spt_with_setup = A.evaluate_order_with_setup(linst.p, linst.family, linst.setup, a.spt.order)
    opt_with_setup = A.brute_force_optimal_with_setup(linst.p, linst.family, linst.setup) if n_jobs <= C.BRUTE_FORCE_MAX_N else None

# --- SPT in Aktion ---------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 SPT in Aktion")
STEP_LABELS = {1: "1 · Aufträge", 2: "2 · Einplanen", 3: "3 · Ergebnis"}
if "spt_step" not in st.session_state or st.session_state.get("spt_step_owner") != data_key:
    st.session_state["spt_step"] = 1
    st.session_state["spt_step_owner"] = data_key
step = st.select_slider("Schritt", options=list(STEP_LABELS), key="spt_step", format_func=lambda s: STEP_LABELS[s])

if step == 2:
    it_col, itplay_col = st.columns([5, 2])
    with it_col:
        upto = st.slider("Eingeplante Aufträge", 1, int(n_jobs), value=int(n_jobs), key="spt_upto")
else:
    upto = int(n_jobs)

view_slot = st.empty()
with view_slot.container():
    if step == 1:
        st.markdown(f"**{n_jobs} Aufträge, unsortiert** (Bearbeitungszeit in Minuten)")
        st.bar_chart({"Bearbeitungszeit": p.tolist()})
    elif step == 2:
        st.markdown(f"**SPT-Reihenfolge nach {upto} von {n_jobs} Aufträgen**")
        st.plotly_chart(build_schedule(p, a.spt.order, upto=upto), width="stretch", key=f"s2_sched_{upto}")
        st.caption(f"Σ Fertigstellungszeiten bisher: {_fmt_int(np.cumsum(p[a.spt.order])[:upto].sum())}")
    else:
        st.markdown("**SPT gegen die längste-zuerst-Reihenfolge (LPT): Σ Fertigstellungszeiten über die Zeit**")
        st.plotly_chart(build_completion_curve(p, a.spt.order, a.lpt.order), width="stretch", key="s3_curve")

if step == 1:
    st.caption(f"Bearbeitungszeiten zwischen {int(p.min())} und {int(p.max())} Minuten (Seed {seed}).")
elif step == 2:
    st.caption("Jeder Balken ist ein Auftrag; die Höhe der Maschinenzeile bleibt gleich, nur die Breite (Bearbeitungszeit) und die Reihenfolge ändern sich.")
else:
    st.caption(f"SPT: Σ Fertigstellungszeiten {_fmt_int(a.spt.total)}. LPT: {_fmt_int(a.lpt.total)} ({a.gap_lpt:.1f} % mehr).")

st.markdown("---")

# --- Ergebnis -------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Sortierung bringt")
st.caption("**Abstand:** Σ Fertigstellungszeiten einer Reihenfolge gegenüber SPT in Prozent. SPT selbst ist deterministisch (kein Zufall im Kern) - nur die Zufalls-Vergleichsreihenfolge streut.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("SPT (Σ Fertigstellungen)", _fmt_int(a.spt.total), help="Die Zielgröße: Summe aller Fertigstellungszeiten in SPT-Reihenfolge.")
m2.metric("LPT (längste zuerst)", f"+{a.gap_lpt:.1f} %", delta_color="off", help="Das genaue Gegenteil von SPT - so schlecht kann eine Reihenfolge werden.")
m3.metric(f"Zufällige Reihenfolge (Mittel über {a.random_runs})", f"+{a.gap_random:.1f} %", delta_color="off", help="Mittel über mehrere zufällige Reihenfolgen derselben Instanz.")
if a.optimal is not None:
    m4.metric("Vollaufzählung (Gegenprobe)", "trifft SPT exakt" if a.spt_matches_optimum else "WEICHT AB", delta_color="off",
              help=f"Alle {n_jobs}! Reihenfolgen durchprobiert - unabhängige Bestätigung, dass SPT wirklich das Minimum ist.")
else:
    m4.metric("Vollaufzählung", f"erst ab n ≤ {C.BRUTE_FORCE_MAX_N}", delta_color="off", help="Bei dieser Größe wäre die Vollaufzählung zu langsam - siehe das Timing-Experiment unten.")

if a.optimal is not None and not a.spt_matches_optimum:
    st.error("⚠️ SPT weicht von der Vollaufzählung ab - das wäre ein Fehler im Beweis oder in der Implementierung, bitte melden.")
else:
    st.success(f"✅ SPT ist {a.gap_lpt:.1f} % besser als die schlechteste Reihenfolge (LPT) und {a.gap_random:.1f} % besser als eine zufällige - bei dieser Zielfunktion beweisbar die beste überhaupt.")

if vehicle == "logistik":
    st.markdown("**Auf dem Werkstatt/Logistik-Vehikel** (Rüstzeit je Familienwechsel berücksichtigt):")
    lm1, lm2 = st.columns(2)
    lm1.metric("SPT, Rüstzeiten mitgerechnet", _fmt_int(spt_with_setup.total), help="Dieselbe SPT-Reihenfolge wie oben, aber die Fertigstellungszeiten berücksichtigen jetzt die Rüstzeit beim Familienwechsel.")
    if opt_with_setup is not None:
        gap = 100.0 * (spt_with_setup.total - opt_with_setup.total) / opt_with_setup.total
        lm2.metric("Echtes Optimum MIT Rüstzeiten", _fmt_int(opt_with_setup.total), delta=f"SPT ist {gap:.1f} % darüber", delta_color="off",
                   help="Vollaufzählung, die die Rüstzeiten selbst mit optimiert - nur für kleine n möglich.")
        if gap > 0.5:
            st.warning(f"⚠️ SPT ist hier NICHT mehr optimal: {gap:.1f} % über dem echten Optimum. Der Beweis oben setzt keine Rüstzeiten voraus - siehe 🚧 unten.")
        else:
            st.info("ℹ️ Bei dieser Instanz liegt SPT trotz Rüstzeiten sehr nah am Optimum - das ist nicht garantiert, siehe die Messreihe unten.")
    else:
        lm2.metric("Echtes Optimum MIT Rüstzeiten", f"erst ab n ≤ {C.BRUTE_FORCE_MAX_N}", delta_color="off")

st.markdown("---")

# --- Sweeps -----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt der Vorsprung von der Instanz ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
if st.button("Sweep über 5 feste Instanzen berechnen (dauert wenige Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {sweep_param}
if sweep_param in st.session_state.get("sweep_done", set()):
    rows_sweep = _sweep(sweep_param, Settings())
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")
    st.caption("Mittel über 5 feste Instanzen (Seeds 100000–100004) mit je drei Zufalls-Ketten für die Vergleichsreihenfolge.")

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Stimmt der Beweis wirklich? Vollaufzählung gegen SPT")
if st.button("Vollaufzählung über n = 2 bis 9 berechnen (dauert etwa 5 Sekunden)", key="opt_start"):
    st.session_state["opt_on"] = True
if st.session_state.get("opt_on"):
    rows_opt = _optimality()
    st.table({"Aufträge": [r["value"] for r in rows_opt], "Trefferquote": [f"{r['match_rate']:.0%}" for r in rows_opt]})
    st.caption("Für jede Instanzgröße 5 feste Instanzen: SPT gegen die Vollaufzählung aller n! Reihenfolgen. Jede Abweichung von 100 % wäre ein Fehler im Beweis oder in der Implementierung.")

st.markdown("---")

st.subheader("🔬 Wie teuer ist eine Vollaufzählung wirklich?")
if st.button("Rechenzeit für n = 2 bis 9 messen (dauert etwa 1 Sekunde)", key="timing_start"):
    st.session_state["timing_on"] = True
if st.session_state.get("timing_on"):
    rows_t = _timing()
    st.plotly_chart(build_timing(rows_t), width="stretch", key="timing_chart")
    last = rows_t[-1]
    st.caption(f"Bei {last['value']} Aufträgen braucht die Vollaufzählung bereits {last['brute_force_seconds']*1000:.0f} ms, SPT {last['spt_seconds']*1000:.3f} ms - {last['brute_force_seconds']/max(last['spt_seconds'],1e-9):.0f}-mal langsamer. n! wächst schneller als jede Potenz von n; n log n praktisch flach im Vergleich.")

st.markdown("---")

st.subheader("🔬 Werkstatt/Logistik: bleibt SPT gut, wenn Rüstzeiten dazukommen?")
if st.button("Rüstzeit von 0 bis 60 Minuten durchfahren (dauert wenige Sekunden)", key="setup_start"):
    st.session_state["setup_on"] = True
if st.session_state.get("setup_on"):
    rows_s = _setup_gap_sweep(min(int(n_jobs), C.BRUTE_FORCE_MAX_N), int(n_families))
    st.plotly_chart(build_setup_gap(rows_s), width="stretch", key="setup_chart")
    st.caption("SPT sortiert weiterhin nur nach Bearbeitungszeit und ignoriert die Rüstzeit beim Familienwechsel; verglichen mit der echten Optimallösung MIT Rüstzeiten (Vollaufzählung, deshalb kleine Instanz). Bei Rüstzeit 0 fallen beide exakt zusammen.")

st.markdown("---")

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Bearbeitungszeit hängt nicht von der Reihenfolge ab** | Sobald Rüstzeiten zwischen Auftragsfamilien dazukommen (Vehikel „Werkstatt/Logistik“), ist SPT nicht mehr beweisbar optimal - der Abstand zum echten Optimum wächst mit der Rüstzeit (siehe Experiment oben). | Kein direkter Nachfolger in dieser Linie; dieselbe Familie von Problemen wie ATCS in `warehouse-transfer-demo` |
| **Alle Aufträge sind gleich wichtig** | Mit unterschiedlichen Gewichten ist SPT nicht mehr optimal - die richtige Regel gewichtet Bearbeitungszeit UND Gewicht. | **WSPT / Smith's Rule** (Folgestück) |
| **Es gibt keine Fristen** | Fristen ändern die Zielfunktion komplett - „möglichst früh fertig“ wird zu „möglichst wenige/kurze Verspätungen“. | **EDD, Moore-Hodgson** (Folgestücke) |
| **Es gibt nur eine Maschine** | Mit mehreren Maschinen wird aus einer Sortierfrage eine Zuordnungs- UND Reihenfolgefrage. | **Johnson-Regel (2 Maschinen), LPT (parallele Maschinen), Job Shop** (Folgestücke) |
"""
)
st.caption(
    "Erstes Stück der Linie „Klassische Scheduling-Theorie“: jedes Folgestück hebt genau EINE dieser Annahmen auf. "
    "Verwandt in anderen Linien: `doppelspiel-demo` (Johnson-Regel praktisch, Kran-Doppelspiel) und "
    "`warehouse-transfer-demo` (SPT/ATCS praktisch, parallele Maschinen mit Fristen)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem** ($1||\sum C_j$): $n$ Aufträge mit Bearbeitungszeit $p_j$ auf einer Maschine; eine Reihenfolge $\pi$ legt
die Fertigstellungszeit $C_j = \sum_{k: \pi(k) \le \pi(j)} p_k$ jedes Auftrags fest. Gesucht: $\pi$, das
$\sum_j C_j$ minimiert.

**Satz.** SPT (aufsteigend nach $p_j$ sortieren) minimiert $\sum_j C_j$.

**Beweis (Vertauschungsargument).** Sei $\pi$ eine beliebige Reihenfolge, in der zwei benachbarte Aufträge $i$
vor $j$ mit $p_i > p_j$ stehen (existiert, solange $\pi$ nicht SPT-sortiert ist). Vertauscht man $i$ und $j$,
ändern sich nur ihre beiden Fertigstellungszeiten - alle anderen bleiben gleich, weil die Summe der
Bearbeitungszeiten VOR den beiden gleich bleibt. Vorher: $C_i = t + p_i$, $C_j = t + p_i + p_j$ (mit $t$ = Summe
davor). Nachher (j vor i): $C_j' = t + p_j$, $C_i' = t + p_j + p_i$. Die Summe sinkt um
$(C_i + C_j) - (C_i' + C_j') = p_i - p_j > 0$. Jede nicht SPT-sortierte Reihenfolge lässt sich also durch eine
solche Vertauschung verbessern; SPT ist der einzige Zustand, an dem keine benachbarte Vertauschung mehr hilft -
und damit das globale Minimum (die Zielfunktion ist eine Summe über alle Paare benachbarter Vertauschungen).

**Kennzahl.** Abstand zu SPT $= 100 \cdot (\text{Summe} - \text{Summe}_{\text{SPT}}) / \text{Summe}_{\text{SPT}}$.
Für $n \le 9$ zusätzlich die Vollaufzählung aller $n!$ Reihenfolgen als unabhängige Gegenprobe.

**Grenzen.** (1) Rüstzeiten zwischen Auftragsfamilien verletzen die Voraussetzung „Bearbeitungszeit hängt nicht
von der Reihenfolge ab“ - SPT bleibt dann nur noch eine gute Heuristik, kein Beweis mehr (Vehikel B). (2) Ohne
Gewichte/Fristen ist die Zielfunktion die einfachste ihrer Art - jedes Folgestück verallgemeinert genau eine
dieser Annahmen.

Implementiert in `spt_algorithm.py` (SPT, Brute-Force-Gegenprobe, Rüstzeit-Variante), `spt_scenario.py`/
`spt_scenario_logistik.py` (die zwei Vehikel), `spt_evaluation.py` (Kennzahlen, Sweeps, Optimalitäts- und
Timing-Messreihe, Vehikel-B-Härtetest).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
