# PeakGuard App Guide v4.0

## 🎨 Neues Design

Die App wurde komplett im PeakGuard Corporate Design überarbeitet:

### Design-Features
- **PeakGuard-Farben**: Dunkelblau (#0f1729) + Hellblau (#0da2e7)
- **Moderne Cards**: Abgerundete Ecken, Schatten, hover-Effekte
- **Icons überall**: Emojis für bessere Orientierung
- **Responsive Layout**: Optimiert für Desktop & Tablet
- **Expander**: Weniger Clutter, wichtige Infos oben

---

## 📂 Workflow

### 1. CSV hochladen
- **Drag & Drop** oder Dateiauswahl
- Unterstützt: 15-min RLM, 1-min Daten, SmartPi, etc.
- **Import-Einstellungen** in Expander (optional):
  - Trennzeichen (Auto-Erkennung)
  - Encoding (UTF-8, CP1252, Latin1)
  - Zahlenformat (Deutsch/Englisch)

### 2. Beispiel-Report (NEU!)
- **Rechts oben**: Beispiel-Report-Button
- Wähle Profil: Lite / Standard / Pro
- Generiert Report mit Dummy-Daten
- Perfekt zum Ausprobieren ohne eigene Daten

### 3. Metadaten
- Standort, Quelle, Zählertyp
- Datenqualität-Status

### 4. Spalten-Mapping
- **Zeitstempel**: Auto-Erkennung von "date", "timestamp", etc.
- **Leistung**:
  - Single-Spalte (Gesamt-P) ODER
  - 3 Spalten (Phase 1/2/3)
- **cosϕ** (optional): Für Blindleistungs-Analyse

### 5. Report-Konfiguration ⭐
**Neue 2-Spalten-Layout:**

**Links: Profil-Auswahl**
- ○ Lite (2-4 Seiten)
- ⦿ Standard (6-10 Seiten) ⭐ **Empfohlen**
- ○ Pro (10-16 Seiten)

**Rechts: Cap-Paket**
- ○ Bronze (P95)
- ○ Silber (P90)
- ○ Gold (P85)
- ○ Manuell

### 6. Download
- **Dynamischer Dateiname**: `2026-01-24-PeakGuard-Report-Standard.pdf`
- Format: `YYYY-MM-DD-PeakGuard-Report-{Profil}.pdf`
- Großer Download-Button (grün)

---

## 🎯 Profil-Unterschiede (Quick Reference)

| Feature | Lite | Standard | Pro |
|---------|------|----------|-----|
| Executive Summary | ✅ | ✅ | ✅ |
| KPI-Kacheln (Seite 1) | ✅ | ✅ | ✅ |
| Top-Peaks | 10 | 10 | 20 |
| Szenarien (Bronze/Silber/Gold) | ❌ | ✅ Cards | ✅ Cards |
| Heatmap + Monatsbalken | ✅ | ✅ | ✅ |
| Peak-Cluster Analyse | ❌ | ✅ | ✅ |
| Roadmap (Top-5-Maßnahmen) | ❌ | ✅ | ✅ |
| Phasen-Unwucht | ❌ | ✅ | ✅ |
| Blindleistung (BLK) | ❌ | ✅ | ✅ |
| **Peak-Kontext (12h/3d)** | ❌ | ❌ | ✅ |
| Glossar | ❌ | ✅ | ✅ |
| **Seiten** | 2-4 | 6-10 | 10-16 |

---

## 💡 Tipps

### CSV-Vorbereitung
1. **Zeitstempel**: ISO-Format bevorzugt (`2024-01-15 14:30:00`)
2. **Header-Zeile**: Spaltennamen in erster Zeile
3. **Konsistenz**: Keine leeren Zeilen zwischen Daten
4. **Encoding**: Bei Umlauten → UTF-8

### Profil-Wahl
- **Lite**: Für Lead-Magnets, Schnell-Checks
- **Standard**: Für die meisten Kunden (beste Balance)
- **Pro**: Für Detailanalysen bei hohem Potenzial

### Cap-Wahl
- **Bronze (P95)**: Konservativ, 5% der Zeit über Cap
- **Silber (P90)**: Ausgewogen, 10% über Cap
- **Gold (P85)**: Aggressiv, 15% über Cap
- **Manuell**: Eigener Zielwert (z.B. 75 kW)

---

## 🚀 Shortcuts

| Aktion | Shortcut |
|--------|----------|
| Beispiel-Report | Button rechts oben |
| Import-Einstellungen | Expander aufklappen |
| Profil wechseln | Radio-Buttons Sektion 5 |
| PDF herunterladen | Großer grüner Button |

---

## 🔧 Troubleshooting

### Problem: CSV wird nicht erkannt
**Lösung:**
1. Expander "Import-Einstellungen" öffnen
2. Trennzeichen manuell wählen
3. Encoding anpassen (oft CP1252 bei Windows-Exports)

### Problem: "Keine Leistungsspalte gefunden"
**Lösung:**
1. In "Rohdaten" prüfen, wie Spalten heißen
2. Korrekte Spalte in Sektion 3 auswählen
3. Bei 3 Phasen: Alle 3 Spalten auswählen

### Problem: Report zu lang (>20 Seiten)
**Lösung:**
1. Profil auf "Standard" oder "Lite" wechseln
2. Intelligente Trigger deaktivieren weniger relevante Module

### Problem: Einsparung = 0 €
**Lösung:**
1. Tarife prüfen (Sektion 4)
2. Peak liegt eventuell schon unter Cap
3. Anderes Cap-Paket wählen (z.B. Gold statt Bronze)

---

## 📊 Beispiel-Workflow (5 Minuten)

1. **App starten**: `./start_report.command`
2. **Beispiel-Report**: Klick oben rechts → "Standard" → Generieren
3. **Eigene Daten**: CSV hochladen
4. **Quick-Config**:
   - Standard-Profil lassen
   - Bronze/Silber wählen
   - Rest auf Auto
5. **Generate**: Button → 2-3 Minuten warten
6. **Download**: PDF mit Datum im Namen

---

## 🎨 Design-Referenz

### Farben
- **Primary**: #0f1729 (Dunkelblau)
- **Accent**: #0da2e7 (Hellblau)
- **Success**: #28A745 (Grün)
- **Gray**: #6C757D

### Icons (Emojis)
- ⚡ PeakGuard Logo
- 📂 Upload
- 🔧 Einstellungen
- 📋 Metadaten
- 💰 Tarife
- 🎯 Ziel
- 🚀 Generieren
- 📥 Download
- 📄 Report
- ✅ Success

---

**Version:** 4.0
**Letzte Aktualisierung:** Januar 2025
