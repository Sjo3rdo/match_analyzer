# Match Analyzer

Een lokale app voor je laptop die voetbalwedstrijden analyseert tot op het niveau van de
individuele speler, in de stijl van Veo. Het verschil met Veo is dat je hier gewone
telefoonvideo's gebruikt die je achteraf uploadt, gefilmd vanaf de zijlijn of vanaf een hoger punt
en gewoon uit de hand.

Alles draait op je eigen laptop. Er gaat geen video naar internet.

## Wat het doet

| Onderdeel | Wat je krijgt |
|---|---|
| Spelers volgen | Iedereen op het veld wordt in elk frame gevonden (YOLO) en over de tijd gevolgd. |
| Teams | Automatisch ingedeeld op shirtkleur. Scheidsrechter en keepers komen bij 'overig'. |
| Wie is wie | Rugnummers lezen (optioneel), en spelers koppelen door ze in de video aan te klikken. |
| Per speler | Gelopen afstand, topsnelheid, sprints, minuten, heatmap en gemiddelde positie. |
| Bal | Balbezit per team, passes, balverlies, passnetwerk. |
| Minimap | Een 2D-bovenaanzicht dat met de video meeloopt. |
| Knippen | Een lange video vóór de analyse in delen knippen (1e/2e helft), warming-up en rust eruit. |
| Clips en delen | Clips maken zoals in Veo: begin/eind, label, spelers taggen, opmerking, tekenen op beeld (beeld bevriest), spotlight die een speler volgt, afspeellijst per speler. Export als mp4, reel of zip, en direct delen via AirDrop, WhatsApp, Berichten of Mail. |

## Installeren en starten (MacBook met Apple Silicon)

Eenmalig nodig: Python 3.10 of nieuwer (`brew install python`).

```bash
git clone https://github.com/sjo3rdo/match_analyzer.git
cd match_analyzer
./run.sh
```

De eerste keer installeert `run.sh` alles in een eigen map (`.venv`). Dat duurt een paar
minuten. Daarna opent de app zich in je browser op http://127.0.0.1:8000. Het detectiemodel
(ca. 40 MB) wordt bij de eerste analyse automatisch gedownload.

Wil je ook rugnummers automatisch laten lezen? Installeer dan:

```bash
.venv/bin/pip install -r requirements-ocr.txt
```

Alle gegevens (video's, analyses, exports) staan in de map `data/`.

## Werkwijze, stap voor stap

Zie het als een puzzel in drie lagen: eerst *zien* (wie staat waar in beeld), dan *plaatsen*
(waar is dat op het echte veld) en dan *benoemen* (wie is dat).

1. **Video's.** Maak een wedstrijd aan en upload je video's. Dat mogen er meerdere zijn,
   bijvoorbeeld per helft of als de telefoon tussendoor stopte. Geef bij elke video de helft en
   de beginminuut op, en klik op **Analyseer**. De app maakt eerst een afspeelbare kopie en zoekt
   daarna ongeveer 10 keer per seconde naar spelers en de bal. Op een MacBook (M-chip) duurt dat
   ongeveer even lang als de video zelf.

2. **Kalibratie.** Je camera ziet het veld schuin, en daardoor lijken spelers die verder weg staan
   kleiner en dichter bij elkaar. Met kalibratie leg je als het ware een doorzichtige plattegrond
   over het beeld:
   - Klik in het beeld op een herkenbaar punt, zoals een hoek van het strafschopgebied, de
     middenstip of het punt waar de middenlijn de zijlijn raakt.
   - Klik daarna hetzelfde punt aan op de veldtekening.
   - Doe dit voor minstens 4 punten, liefst 6 of meer, verspreid over het beeld. De witte lijnen
     laten zien of het klopt.
   - **Filmen vanaf de zijlijn op ooghoogte?** Dan zie je vaak maar 1 of 2 hoekpunten. Klik dan ook
     op een plek **ergens op een veldlijn**, zoals de zijlijn vlak voor je, de 16-meterlijn of de
     doellijn, en daarna op die lijn in de veldtekening (hij kleurt rood). Een punt telt 2, een lijn
     telt mee met hoogstens 2 punten, en je hebt er samen 8 nodig, bijvoorbeeld 1 punt en 3 lijnen.
     Let op: precies 2 punten en 2 lijnen ligt wiskundig niet vast; voeg dan nog iets toe.
   - **Vertel de app waar je stond.** Klik bij **📍 Waar stond je bij het filmen?** op de veldtekening
     waar je ongeveer stond, en kies je hoogte (staand, heuvel of tribune). Dan rekent de app met een
     cameramodel en is **1 punt + 1 lijn** al genoeg, bijvoorbeeld een doelpaal en de zijlijn voor je.
     Nog beter is 2 punten + 1 lijn of 1 punt + 2 lijnen; dan rekent de app ook uit of je hebt ingezoomd.
   - **Zoek via GPS.** iPhone-video's bevatten meestal de GPS-positie. De knop **📡 Zoek via GPS**
     zoekt het voetbalveld op in OpenStreetMap en zet je positie automatisch op de tekening. Daarvoor
     wordt alleen de coördinaat naar OpenStreetMap gestuurd, en alleen als jij op de knop klikt. De
     iPhone slaat de positie op ongeveer 5 à 10 m nauwkeurig op; klik je plek gerust preciezer aan.
   - Omdat je uit de hand filmt, voeg je elke 30 tot 60 seconden een *sleutelframe* toe, en ook na
     elke flinke zwenk of zoom. Tussen de sleutelframes volgt de app de camerabeweging zelf, zoals
     een lijm die de plattegrond op het beeld vasthoudt. Na de analyse voorspelt de app de punten
     (de gele stippellijnen), zodat je ze alleen nog hoeft bij te schuiven.

3. **Spelers.** Voer de selectie in, met rugnummers. Klik op **Automatisch koppelen** om
   tracks met een leesbaar rugnummer te koppelen. De rest koppel je via de kaartjes of door in
   **Video + minimap** te pauzeren en op een speler te klikken.

4. **Bekijken.** Bij **Statistieken** zie je de team- en spelerscijfers, heatmaps, de teamvorm
   en het passnetwerk, en kun je alles als CSV exporteren.

5. **Clips en delen.** Maak bij **Video + minimap** met één klik een clip van het moment dat je
   ziet, of zet bij **Clips & delen** een automatische suggestie (sprint, pass) om in een clip. Daar
   kun je:
   - begin en eind per seconde verschuiven;
   - een label, spelers en een opmerking toevoegen;
   - een **spotlight** op een speler zetten: een gele ring onder zijn voeten met zijn naam, die hem
     door de clip volgt;
   - op het beeld **tekenen**: pijlen, lijnen, cirkels, vrije lijnen en tekst. In de export bevriest
     het beeld dan een paar seconden met de tekening erop, zoals bij een tv-analyse;
   - alle clips van één speler als **afspeellijst** afspelen of als reel exporteren.

   Met **Exporteer** maak je een mp4 (1080p). Daarna kun je op **Deel** klikken voor AirDrop,
   WhatsApp, Berichten of Mail; dat werkt in Safari. Of je downloadt het bestand.

**Een lange video knippen.** Klik bij **1. Video's** op **Knippen**. Klik **Begin deel** bij de
aftrap en **Einde deel** bij het eindsignaal, en doe hetzelfde voor de tweede helft. Alles daartussen
(warming-up, rust) valt weg. Het knippen is binnen seconden klaar en kost geen kwaliteit. Een deel kan
wel tot ongeveer 1 seconde eerder beginnen dan je koos.

## Tips voor het filmen

- **Hoog is beter.** Vanaf een tribune, een heuvel of een ladder zie je meer veld en staan spelers
  minder voor elkaar.
- **Liggend filmen**, in 1080p of 4K, met 30 of 60 fps.
- **Rustig zwenken.** Snelle rukken en veel in- en uitzoomen maken de kalibratie lastiger.
- **Houd veldlijnen in beeld.** Die heeft de kalibratie nodig.
- **Vanaf de zijlijn op ooghoogte** werkt het ook, maar spelers aan de overkant zijn dan klein en
  afstanden daar minder nauwkeurig. Houd liefst de zijlijn voor je én een doel of de 16-meterlijn
  in beeld.

## Hoe nauwkeurig is het?

Wees realistisch: met telefoonbeelden vanaf de zijlijn kom je niet aan de nauwkeurigheid van
een vaste 180°-camera.

- **Afstand en snelheid.** Hoe dichter bij de camera, hoe beter. Aan de verre kant kan een fout
  van 1 tot 2 m per positie ontstaan. De app strijkt posities glad en negeert onmogelijke sprongen.
- **Bal.** Een kleine, snelle bal is vaak niet te zien. Balbezit en passes zijn daarom een
  *schatting*.
- **Rugnummers.** Vanaf de zijlijn zijn die vaak onleesbaar. Gebruik ze daarom als suggestie en
  controleer de koppelingen.
- **Spelers buiten beeld.** Wie buiten beeld is, wordt niet gemeten. Het aantal minuten laat zien
  hoe lang iemand in beeld was.

## Techniek

- **Backend.** Python, FastAPI en SQLite (`app/`).
- **Detectie.** Ultralytics YOLO11, op de Apple-GPU via MPS (`app/detection.py`).
- **Volgen.** Een eigen ByteTrack-achtige tracker die camerabeweging compenseert
  (`app/tracking.py`).
- **Camerabeweging en kalibratie.** Optical flow op de achtergrond met homografieën. Tussen
  sleutelframes wordt gemengd om drift te beperken (`app/calibration.py`).
- **Teams.** K-means op shirtkleur in de Lab-kleurruimte, waarbij graspixels worden weggefilterd
  (`app/teams.py`).
- **Statistieken.** Te vinden in `app/analytics.py`.
- **Interface.** Gewone HTML en JavaScript zonder build-stap (`static/`).

Instellingen, zoals het model, de analyse-fps en de sprintgrens, staan in `app/config.py`. Je
kunt het model ook kiezen met een omgevingsvariabele, bijvoorbeeld
`MATCH_ANALYZER_MODEL=yolo11l.pt ./run.sh` voor meer precisie (langzamer).

### Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

De end-to-end-test maakt een synthetische wedstrijdvideo en doorloopt de hele keten: uploaden,
kalibreren, analyseren, koppelen, statistieken en export.
