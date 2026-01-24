# PeakGuard Report v4.0 - Changelog

## 🎯 Übersicht
Version 4.0 ist ein umfassendes Redesign mit Fokus auf **Kompaktheit**, **Modernität** und **Intelligenz**.

---

## 🆕 Neue Features

### 1. Profile-System (Lite / Standard / Pro)
**3 Report-Varianten für unterschiedliche Bedürfnisse:**

- **Lite (2-4 Seiten)**: Einstieg / Lead Magnet
  - Executive Summary
  - Top-10 Peaks
  - Heatmap + Monatsbalken

- **Standard (6-10 Seiten)**: Empfohlen, Kernprodukt
  - Alles von Lite
  - Bronze/Silber/Gold Szenarien (als Cards)
  - Peak-Cluster kompakt
  - Maßnahmen-Roadmap
  - Glossar

- **Pro (10-16 Seiten)**: Detailanalyse
  - Alles von Standard
  - Peak-Kontext (12h/3d-Fenster für Top-3-Peaks)
  - Top-20 Peaks
  - Alle Detail-Module

### 2. Executive Summary (Seite 1)
**Neue erste Seite mit "60-Sekunden-Einstieg":**
- 3 KPI-Kacheln (Peak, Einsparung, Problemtyp)
- Top-3-Hebel als Action-Cards
- Data-Quality-Box
- Sofort verständlich ohne Vorkenntnisse

### 3. Moderne Visualisierungen
**Neue Charts:**
- **Monatsbalken**: Zeigt Peaks pro Monat (Balken = Überschreitungen, Linie = Max-Peak)
- **Peak-Kontext** (Pro): 12h + 3d-Fenster um Top-3-Peaks
- Alle Charts haben jetzt **Captions** (1-Satz-Interpretation)

**Optimierte Charts:**
- Heatmap: kompakter, bessere Farbskala
- Zeitreihe: klarere Überschreitungsmarkierung
- Events-Scatter: verbesserte Lesbarkeit

### 4. Card-basiertes Design
**Ersetzt Tabellen-Optik durch moderne Cards:**
- KPI-Kacheln statt Zeilen
- Szenarien (Bronze/Silber/Gold) als farbcodierte Cards
- Action-Cards für Top-3-Hebel
- Mehr Weißraum, bessere Lesbarkeit

### 5. Intelligente Trigger
**Report passt sich automatisch an:**
- Peak-Kontext nur bei Potenzial > 5000 €/a ODER > 20 Events
- BLK-Modul nur wenn cosϕ-Daten vorhanden
- Phasen-Modul nur wenn 3-Phasen-Daten vorhanden
- Spart unnötige Seiten bei geringer Relevanz

### 6. Glossar "So lesen Sie den Report"
**Neue 1/2-Seite Begriffserklärung:**
- 15-min-Mittelwert
- Cap / Peak-Shaving
- Benutzungsstunden
- Verschiebbare kWh
- P95/P90/P85
- Peak-Problemtyp

Reduziert Rückfragen um geschätzt 50%.

### 7. "Was heißt das praktisch?"
**Interpretation nach jeder Analyse:**
- Peak-Cluster: Konkrete Maßnahmen-Hinweise
- Phasen-Unwucht: Elektrik-Empfehlungen
- BLK: Wirtschaftlichkeits-Einordnung

### 8. Parameter-Config-System
**Zentrale Konfiguration für Schwellenwerte:**
```python
@dataclass
class ReportConfig:
    peak_context_min_savings: float = 5000.0  # €/a
    peak_context_min_events: int = 20
    unbalance_threshold_kw: float = 3.0
    blk_cosphi_threshold: float = 0.9
    top_peaks_lite: int = 10
    top_peaks_standard: int = 10
    top_peaks_pro: int = 20
```

Später erweiterbar auf YAML/JSON-Import.

---

## 🎨 Design-Verbesserungen

### Design-Token-System
**Konsistente Spacing/Fonts:**
- `DesignTokens.SPACE_XS` bis `SPACE_XXL` (2mm - 16mm)
- `FONT_SIZE_XXS` bis `FONT_SIZE_HUGE` (7pt - 24pt)
- `CARD_PADDING`, `GUTTER`, `COL_2/COL_3` für Layout

### Moderneres Layout
- **Mehr Weißraum**: Durchschuss erhöht, Padding optimiert
- **Klarere Hierarchie**: 3-stufiges Heading-System
- **Weniger Linien**: Dezente Borders statt harter Rahmen
- **Farbcodierung**: Bronze/Silber/Gold, Prioritäten (Quick Win/Investition)

### Chart-Theme v2
- Dünnere Gridlines (0.4 statt 0.5)
- Dickere Datenlinien (2.5 statt 2.0)
- Keine Top/Right-Spines
- Konsistente Fonts über alle Charts

---

## 📊 Funktionale Verbesserungen

### Peak-Problemtyp-Erkennung
**Automatische Klassifizierung:**
- **Kurzspitzen**: >60% Peaks ≤15 min → "Gleichzeitigkeit vermeiden"
- **Langspitzen**: >40% Peaks ≥60 min → "Grundlast/Prozess optimieren"
- **Gemischtes Muster**: sonst

### Peak-Diagnose (Pro)
**Für Top-3-Peaks:**
- Gleichzeitigkeit (kurzer Peak über Umfeld)
- Dauerlast (lange Überschreitung ≥2h)
- Anfahrvorgang (starke Schwankungen)
- Normaler Lastgang

### Szenarien-Optimierung
**Cards statt Tabelle:**
- Übersichtlicher
- Einsparung als Eye-Catcher
- Farbcodierung Bronze/Silber/Gold

### Top-Peaks profilabhängig
- Lite/Standard: Top-10 (1 Seite)
- Pro: Top-20 (1-2 Seiten)

---

## 🔧 Technische Verbesserungen

### Modularität
- Profile-Klasse mit Feature-Flags
- Profilabhängige Chart-Generierung
- Intelligente Modul-Aktivierung

### Code-Qualität
- Type Hints durchgängig
- Dataclasses für Struktur
- Zentrale Config-Klasse
- Design-Tokens statt Magic Numbers

### Performance
- Charts nur bei Bedarf generieren
- Profilabhängige Berechnungen
- Intelligentes Caching (temp files)

---

## 🐛 Bugfixes

### v3.1 → v4.0
- Heatmap: Lesbarkeit bei hohen Werten verbessert (weiße Kreise)
- Szenario-Tabelle: Prozent-Reduktion korrekt berechnet
- PageBreaks: Keine "Orphan"-Headlines mehr
- Chart-Captions: Immer vorhanden, nicht mehr abhängig von Chart-Anzahl

---

## 📈 Migration v3.1 → v4.0

### Breaking Changes
**Keine!** v4.0 ist vollständig rückwärtskompatibel.

### Neue Parameter (optional)
```python
build_pdf_report(
    ...
    profile=PROFILE_STANDARD,  # NEU: Lite/Standard/Pro
    # Alle bisherigen Parameter funktionieren weiter
)
```

### Empfohlene Anpassungen
1. **app.py**: Profil-Auswahl hinzufügen (bereits gemacht in diesem Update)
2. **Config**: Bei Bedarf `ReportConfig` anpassen für eigene Schwellenwerte
3. **Farben**: Bei Bedarf `custom_primary/accent` in Config setzen

---

## 📝 Nächste Schritte (Roadmap)

### v4.1 (geplant)
- [ ] YAML/JSON-Config-Import
- [ ] Mehrere Caps parallel testen
- [ ] Benchmark-Vergleich (Branche)
- [ ] Export als PowerPoint (optional)

### v5.0 (Vision)
- [ ] Live-Dashboard (Streamlit)
- [ ] Automatische Maßnahmen-Priorisierung (ML)
- [ ] Multi-Site-Vergleich
- [ ] API-Integration

---

## 🙏 Credits

**v4.0 Development:**
- Redesign basierend auf PRD v2 (kompakt, modern, modular)
- Sprint 1-4 Implementation (Jan 2025)
- PeakGuard CI/CD (Farben unverändert)

**Dependencies:**
- reportlab >= 3.6
- matplotlib >= 3.5
- pandas >= 1.3
- numpy >= 1.21
- streamlit >= 1.20

---

**Version:** 4.0
**Release Date:** Januar 2025
**Status:** Production Ready ✅
