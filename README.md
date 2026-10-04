# Match Analyzer

Een lokale app voor je laptop die voetbalwedstrijden analyseert tot op het niveau van de
individuele speler, in de stijl van Veo. Het verschil met Veo is dat je hier gewone
telefoonvideo's gebruikt die je achteraf uploadt, gefilmd vanaf de zijlijn of vanaf een hoger punt
en gewoon uit de hand.

Alles draait op je eigen laptop. Er gaat geen video naar internet, en ook geen gebruiksgegevens:
de anonieme statistieken die de detectiesoftware (Ultralytics) normaal verstuurt, staan uit.

## Wat het doet

| Onderdeel | Wat je krijgt |
|---|---|
| Spelers volgen | Iedereen op het veld wordt in elk frame gevonden (YOLO) en over de tijd gevolgd. |
| Teams | Automatisch ingedeeld op shirtkleur, ook in zon en schaduw, en gelijk over alle video's van een wedstrijd. Scheidsrechter, keepers en publiek komen bij 'overig'. |
| Wie is wie | Rugnummers lezen (optioneel), spelers koppelen door ze in de video aan te klikken, en een koppel-assistent die de rest van het spoor van een speler voorstelt. Selecties bewaar je en neem je over in de volgende wedstrijd. |
| Per speler | Gelopen afstand, topsnelheid, sprints, minuten, heatmap en gemiddelde positie. |
| Bal | Balbezit per team, passes, balverlies, passnetwerk. |
| Minimap | Een 2D-bovenaanzicht dat met de video meeloopt. |
| Knippen | Een lange video vóór de analyse in delen knippen (1e/2e helft), warming-up en rust eruit. |
| Hoogtepunten | Suggesties uit de analyse (sprints, passes) en uit het geluid (gejuich, fluitsignalen). |
| Clips en delen | Clips maken zoals in Veo: begin/eind, label, spelers taggen, opmerking, tekenen op beeld (beeld bevriest), spotlight die een speler volgt, afspeellijst per speler. Export als mp4, reel of zip, en direct delen via AirDrop, WhatsApp, Berichten of Mail. |

## Installeren en starten (MacBook met Apple Silicon)

Eenmalig nodig: Python 3.10 of nieuwer (`brew install python`).

```bash
git clone https://github.com/sjo3rdo/match_analyzer.git
cd match_analyzer
./run.sh
```

De eerste keer installeert `run.sh` alles in een eigen map (`.venv`). Dat duurt een paar
minuten. Daarna opent de app zich in je browser op http://127.0.0.1:8000, zodra hij klaar is
met opstarten. Het detectiemodel (ca. 40 MB) wordt bij de eerste analyse automatisch gedownload.

**Bijwerken.** Haal de nieuwste versie op met `git pull` en start opnieuw met `./run.sh`. Rechtsboven
in de app staat de versie en de commit (bijv. `versie 0.9 · 7e771cd`); die moet gelijk zijn aan
`git log -1 --oneline`. De browser haalt de interface na een update altijd vers op. Kwam je van een
versie zonder versienummer rechtsboven, druk dan één keer op ⌥⌘R in Safari om de oude, bewaarde
pagina weg te gooien (anders zie je bijv. geen deelknop of krijg je "Method Not Allowed").

Wil je ook rugnummers automatisch laten lezen? Installeer dan:

```bash
.venv/bin/pip install -r requirements-ocr.txt
```

Alle gegevens (video's, analyses, exports) staan in de map `data/`.

## Werkwijze, stap voor stap

Zie het als een puzzel in drie lagen: eerst *zien* (wie staat waar in beeld), dan *plaatsen*
(waar is dat op het echte veld) en dan *benoemen* (wie is dat).

Rechtsboven wissel je met ☀︎/☾ tussen het donkere en het lichte thema. Bovenaan staan de tabbladen. De eerste drie zijn de stappen die je doorloopt: **Video's →
Kalibratie → Spelers**. Een groen vinkje betekent klaar en het gele bolletje is de volgende stap
(houd de muis erop om te zien wat er nog moet). Daarna bekijk je het resultaat bij **Video +
minimap**, **Statistieken** en **Clips & delen**.

1. **Video's.** Maak een wedstrijd aan en upload je video's. Dat mogen er meerdere zijn,
   bijvoorbeeld per helft of als de telefoon tussendoor stopte. Geef bij elke video de helft en
   de beginminuut op, en klik op **Analyseer**. De app maakt eerst een afspeelbare kopie en zoekt
   daarna ongeveer 10 keer per seconde naar spelers en de bal. Je ziet hoe lang het nog duurt.
   - **Nauwkeurig of Snel.** *Nauwkeurig* vindt ook spelers ver weg en de bal het best. *Snel* is
     ongeveer twee keer zo snel, maar mist vaker verre spelers en de bal: handig om eerst snel een
     hele wedstrijd door te nemen.
   - **Richting ⇄ omdraaien.** In de rust wisselen de teams van kant. Met dit vinkje (standaard aan
     bij de 2e helft) spiegelt de app die video in de statistieken, zodat heatmaps en teamvorm over
     de hele wedstrijd kloppen. Ben je in de rust zelf naar de andere kant van het veld gelopen,
     vink het dan uit.

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
   - **Uit de lijst kiezen.** Onder de veldtekening staat een lijst met alle punten en lijnen. Zie je
     alleen een zijlijn en verder geen vast punt (geen hoekvlag, geen middenlijn)? Kies dan
     **Zijlijn onder** (de kant waar jij staat als je onderaan de tekening staat) en klik er in het
     beeld een paar plekken op aan, verspreid over de lijn. Samen met je positie en één ander punt of
     lijn ligt het veld dan vast.
   - **Het doel.** Klik je in de veldtekening bij een doel, dan kies je wat je in het beeld aanklikt:
     de **voet** van een paal (waar hij de grond raakt), de **bovenkant** van een paal (waar de lat
     begint, 2,44 m hoog) of een plek **op de lat**. De bovenkant en de lat hangen in de lucht en
     vertellen de app hoe ver weg het doel is en hoeveel je hebt ingezoomd. Ze tellen mee zodra de app
     weet waar je stond (zie hieronder). Na het klikken tekent de app het doel in het beeld, zodat je
     ziet of het klopt.
   - **Vertel de app waar je stond.** Klik bij **📍 Waar stond je bij het filmen?** op de veldtekening
     waar je ongeveer stond, en kies je hoogte (staand, heuvel of tribune). Dan rekent de app met een
     cameramodel en is **1 punt + 1 lijn** al genoeg, bijvoorbeeld een doelpaal en de zijlijn voor je.
     Nog beter is 2 punten + 1 lijn of 1 punt + 2 lijnen; dan rekent de app ook uit of je hebt ingezoomd.
   - **Veldmaten.** De app gaat uit van 105 × 68 m. Amateurvelden zijn vaak kleiner, bijvoorbeeld
     100 × 64. Vul de echte maten in bij **Veldmaten** (op de pagina Kalibratie); dan kloppen
     afstanden en posities beter. Het strafschopgebied en de middencirkel zijn altijd even groot.
     Na **Zoek via GPS** biedt de app aan om de maten uit OpenStreetMap over te nemen.
   - **Zoek via GPS.** iPhone-video's bevatten meestal de GPS-positie. De knop **📡 Zoek via GPS**
     zoekt het voetbalveld op in OpenStreetMap en zet je positie automatisch op de tekening. Daarvoor
     wordt alleen de coördinaat naar OpenStreetMap gestuurd, en alleen als jij op de knop klikt. De
     iPhone slaat de positie op ongeveer 5 à 10 m nauwkeurig op; klik je plek gerust preciezer aan.
   - **Laat de app het veld zoeken.** Weet de app waar je stond, dan zoekt hij zelf de witte lijnen en
     legt hij de veldtekening erop, zonder klikken (**🤖 Zoek het veld automatisch**; gebeurt vanzelf
     als er nog geen sleutelframe is). Zie het als rondkijken met een plattegrond in je hand tot alle
     lijnen kloppen. Controleer het voorstel: vallen de witte lijnen op het veld? Sleep punten bij waar
     nodig en klik **✓ Klopt**. Zegt de app **twijfel**, dan zijn er te weinig lijnen te zien (bijv.
     alleen de zijlijn); kies dan een moment met de 16-meter, middenlijn of cirkel in beeld. Klik je
     plek zo precies mogelijk aan: een paar meter ernaast maakt het zoeken een stuk lastiger. Alleen de
     GPS-positie (5 à 10 m nauwkeurig) is vaak niet genoeg.
   - **Inzoomen.** Knijp op het trackpad (of ⌥ + scrollen) om in te zoomen, bijvoorbeeld om de verre
     hoekvlag precies aan te klikken. Verschuiven doe je met twee vingers, of Shift + slepen.
     Een punt dat je ingezoomd zet, telt zwaarder mee (tot 4× bij flink inzoomen): de lijn gaat dan
     door jouw precieze stippen, en grovere punten van ver weg geven mee.
   - **Punten bewegen mee met het veld.** Zet je een punt en schuif je daarna naar een ander moment,
     dan schuift het punt mee met de camerabeweging. Zo kun je punten van verschillende momenten in één
     sleutelframe combineren: bijv. de middenstip nu en de verre hoekvlag als de camera daar is.
     Punten die daardoor buiten beeld vallen, tellen gewoon mee.
   - **Eén sleutelframe is genoeg.** Na de analyse volgt de app de camerabeweging, zoals een lijm die
     de plattegrond op het beeld vasthoudt. Elke seconde zoekt hij daarnaast de witte veldlijnen op
     en legt hij de plattegrond er opnieuw precies op (**🤖 Automatisch bijgesteld**). Dit start
     vanzelf zodra je een sleutelframe opslaat, of na de analyse. De gele stippellijnen laten op elk
     moment zien hoe goed het past. Onder **🤖 Automatisch bijgesteld** staat de lijst: klik op een tijd
     om te kijken, ✓ keurt goed, ✗ verwijdert, ✎ zet de punten klaar om zelf bij te stellen, of keur
     ze met **✓ Alles accepteren** in één keer goed. Goedgekeurde blijven staan als je opnieuw bijstelt.
   - Het bijstellen werkt het best als je ook hebt ingesteld **waar je stond**: dan houdt de app je
     zoom en scheefstand vast en kan er niets ongemerkt wegglijden.
   - Kijk je lang langs alleen de zijlijn, zonder dwarslijnen zoals de 16-meterlijn, middenlijn,
     doellijn of cirkel? Dan is niet te zien hoeveel je gedraaid hebt. De app slaat die momenten over
     en overbrugt ze met de camerabeweging tot er weer dwarslijnen in beeld komen. Past het ergens
     niet, zet daar dan zelf een extra sleutelframe.

3. **Spelers.** Toeschouwers, wissels langs de lijn en mensen vlak voor de camera worden zoveel
   mogelijk weggefilterd: wie meestal op dezelfde plek staat (gemeten tegen de achtergrond, dus
   los van het zwenken), of wiens voeten buiten beeld vallen, telt niet als speler. Na de
   kalibratie valt ook iedereen buiten het veld af. Voer de selectie in, met rugnummers.
   - **Kalibratie klopt niet?** Valt bijna iedereen die rondloopt buiten het veld, dan gaat de app
     ervan uit dat de kalibratie niet goed ligt. Hij filtert dan niet op het veld en laat een
     waarschuwing zien met een link naar **Kalibratie**.
   - **Rustig de lijst afwerken.** Een kaartje dat je afhandelt (koppelen, 🚫) verdwijnt meteen, en de
     pagina blijft staan waar je was. Vragen en de assistent verschijnen in een balk onderin.
   - **Toeschouwer weghalen.** Staat er toch iemand van het publiek tussen? Klik op 🚫 op zijn
     kaartje (of kies **Toeschouwer** bij het team, ook in **Video + minimap**). Hij telt dan nergens
     meer mee. De app zoekt meteen naar personen met dezelfde soort kleding op dezelfde plek, ook in
     je andere video's van deze wedstrijd, en vraagt of die ook weg mogen. Vergist? Vink **toon
     weggehaalde** aan en zet het team terug.
   - **Selectie overnemen.** Klik bij een team op **💾 Bewaar als vaste selectie**. Bij een volgende
     wedstrijd kies je die teamnaam bij het aanmaken (of bij **📋 Selectie overnemen**), en staan de
     spelers er meteen in. Je kunt ook de selectie van een eerdere wedstrijd overnemen.
   - **Teams omwisselen.** De app weet niet welke shirtkleur jouw team is. Staat jouw team bij
     "Uit"? Klik dan op **⇄ Teams omwisselen**. Video's van dezelfde wedstrijd houdt de app zelf
     gelijk.
   - **De app leert van je correcties.** Zet je bij een paar spelers zelf het goede team, klik dan
     op **↻ Opnieuw indelen**: de app gebruikt jouw keuzes als voorbeeld voor de rest van de video.
     De teamkleuren die zo ontstaan, onthoudt de app bij de wedstrijd en bij je vaste selectie, zodat
     een volgende video of wedstrijd met hetzelfde tenue meteen goed begint.
   - **Koppel-assistent.** Koppel je een track aan een speler, dan stelt de app de tracks voor die
     waarschijnlijk ook van hem zijn: ze zijn niet tegelijk in beeld, beginnen ongeveer waar het
     vorige stuk ophield en hebben hetzelfde shirt. Zie het als een spoor in de sneeuw dat steeds
     even onderbroken is: heb je één stuk, dan zoekt de app het volgende. Met **✓ Koppel** neem je
     een voorstel over, daarna zoekt de app verder. Via 🔍 achter een speler open je de assistent
     ook zelf. Klik op **Automatisch koppelen** om
   tracks met een leesbaar rugnummer te koppelen. De rest koppel je via de kaartjes of door in
   **Video + minimap** te pauzeren en op een speler te klikken.

   **De bal.** In **Video + minimap** zie je een cirkel om de bal: wit = gevonden, geel gestippeld =
   geschat (de bal was even niet te zien en de app trekt een lijn tussen ervoor en erna), groen = door
   jou aangewezen. Mist de app de bal op een belangrijk moment? Pauzeer, klik **⚽ Bal aanwijzen** en
   klik op de bal. Ziet de app iets anders aan voor de bal, klik dan **🚫 Geen bal hier**. Jouw
   aanwijzingen gaan altijd voor en worden gebruikt voor balbezit en passes.

4. **Bekijken.** Bij **Statistieken** zie je de team- en spelerscijfers, heatmaps, de teamvorm
   en het passnetwerk, en kun je alles als CSV exporteren.

5. **Clips en delen.** Maak bij **Video + minimap** met één klik een clip van het moment dat je
   ziet (6 s ervoor tot 4 s erna; hij verschijnt meteen onder "Momenten in deze video" en bij
   **Clips & delen**), of zet bij **Clips & delen** een automatische suggestie om in een clip: een
   sprint, een pass, of iets uit het geluid. **📣 Gejuich** betekent dat het ineens veel luider werd,
   bijvoorbeeld na een goal of grote kans; de clip begint daarom 10 s ervoor. Daarnaast vindt de app
   **🔔 Fluitsignalen**. Bij Clips & delen kun je:
   - begin en eind per seconde verschuiven;
   - een label, spelers en een opmerking toevoegen;
   - een **spotlight** op een speler zetten: een gele ring onder zijn voeten met zijn naam, die hem
     door de clip volgt;
   - op het beeld **tekenen**: pijlen, lijnen, cirkels, vrije lijnen en tekst. In de export bevriest
     het beeld dan een paar seconden met de tekening erop, zoals bij een tv-analyse;
   - alle clips van één speler als **afspeellijst** afspelen of als reel exporteren.

   Met **Exporteer** maak je een mp4 (1080p). Daarna kun je op **Deel** klikken voor AirDrop,
   WhatsApp, Berichten of Mail; dat werkt in Safari. Of je downloadt het bestand.

**Een lange video knippen.** Klik bij **Video's** op **Knippen**. Klik **Begin deel** bij de
aftrap en **Einde deel** bij het eindsignaal, en doe hetzelfde voor de tweede helft. Alles daartussen
(warming-up, rust) valt weg. Het knippen is binnen seconden klaar en kost geen kwaliteit. Een deel kan
wel tot ongeveer 1 seconde eerder beginnen dan je koos.

## De app slimmer maken (trainen)

Zie de app als een stagiair die meekijkt: alles wat jij verbetert, schrijft hij op. Pas als jij op
een knop drukt, gaat hij daarmee oefenen. Dat kost tijd en rekenkracht, dus jij kiest het moment.

1. **Video's klaarzetten.** Klopt de kalibratie in een video (de gele lijnen liggen op de witte
   lijnen) en heb je de bal en toeschouwers waar nodig verbeterd? Klik dan bij **Kalibratie** op
   **✓ Gebruik voor training**. Er gebeurt nog niets; de video staat alleen klaar.
2. **Trainen wanneer het jou uitkomt.** Klik rechtsboven op **🧠 Trainen**. Je ziet hoeveel er
   klaarstaat en hoe lang het ongeveer duurt. Kies **Snel** of **Grondig**:
   - **Spelers en bal herkennen.** Het detectiemodel oefent op je eigen beelden. Vooral de bal: wat
     de app alleen ingezoomd vond of wat jij aanwees, leert hij in het gewone beeld te zien. Op een
     MacBook met Apple-chip is de schatting 5 à 15 minuten per honderd beelden (Snel); de app laat
     vooraf zien hoe lang het ongeveer duurt en past dat tijdens het trainen aan.
   - **Het veld herkennen.** Een klein netwerk leert uit goed gekalibreerde beelden welke pixels
     veldlijn zijn, zodat het automatisch kalibreren ook lukt op velden met slechte lijnen.
   - **Spelers herkennen.** Per speler leert de app hoe hij eruitziet (houding, haar, schoenen,
     kleur) uit de stukken die je aan hem hebt gekoppeld. Dat kan ook per speler: klik op **🧠**
     achter zijn naam bij **Spelers**. Het profiel wordt bewaard bij je vaste selectie, zodat de app
     hem in een volgende wedstrijd zelf voorstelt (**🧠 Herkend** boven de tracks, en in de
     koppel-assistent). Het is een hulp, geen zekerheid: teamgenoten in hetzelfde shirt lijken veel
     op elkaar. Controleer de voorstellen; hoe meer je koppelt, hoe beter het profiel.
3. **Alleen als het beter is.** Na het oefenen vergelijkt de app het nieuwe model met het oude op
   beelden die het niet heeft gezien (ongeveer 1 op de 7 stukjes van 10 seconden houdt de app
   apart). Alleen als het duidelijk beter is, gebruikt de app voortaan het nieuwe. Bij **Eerdere
   trainingen** zie je de cijfers en kun je zelf een versie kiezen, of **↺ Terug naar standaard**.

Tijdens het trainen kun je de app gewoon gebruiken (iets trager). Rechtsboven zie je hoe ver het is;
met **■ Stoppen** breek je het af. Laat de laptop aan de lader en zet hem niet in slaap. Alles blijft
op je eigen laptop.

## Wat de app onthoudt

Zoals een trainer die een veld al kent, hoef je sommige dingen maar één keer in te stellen:

- **Veldmaten per veld.** Stel je de veldmaten in, dan onthoudt de app ze bij de GPS-plek van je
  video's. Film je later weer op dat veld, dan staan de maten er bij het uploaden meteen goed in.
- **Je plek per video.** Klik je in één video aan waar je stond, dan krijgen andere video's van
  dezelfde wedstrijd die vanaf (bijna) dezelfde GPS-plek zijn gefilmd die plek ook. Klik hem gerust
  preciezer aan.
- **Teamkleuren en toeschouwers.** Zie **De app leert van je correcties** en **Toeschouwer
  weghalen** hierboven.

Alles blijft op je eigen laptop.

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
- **Bal.** Een kleine, snelle bal is vaak niet te zien. De app volgt hem daarom als een vogelaar:
  is hij even weg, dan zoomt de app in op de plek waar hij heen ging (uitsnede op volle resolutie),
  en is hij lang weg, dan zoekt de app af en toe het hele beeld in stukken af. Korte gaten (tot
  1 s) vult de app op. Dat maakt de analyse wat langzamer. Balbezit en passes blijven een
  *schatting*; wijs de bal zelf aan op momenten die ertoe doen. Video's die met een eerdere versie
  zijn geanalyseerd, hebben dit pas na **Opnieuw** analyseren.
- **Rugnummers.** Vanaf de zijlijn zijn die vaak onleesbaar. Gebruik ze daarom als suggestie en
  controleer de koppelingen.
- **Teams en publiek.** In fel zonlicht lijkt een donkerblauw shirt soms grijs; zo'n speler kan bij
  'overig' terechtkomen. Toeschouwers die bewegen (bijvoorbeeld langs de lijn lopen) worden pas
  na de kalibratie herkend, omdat ze dan buiten het veld staan.
- **Geluid.** Gejuich en fluitsignalen zijn suggesties. Wind in de microfoon, iemand die vlak
  naast je praat of een losse roep van een speler ("hier!") telt de app niet als gejuich, en per
  halve minuut krijg je hooguit één suggestie van elke soort. Controleer het altijd even: een
  langgerekte roep kan soms als fluitsignaal binnenkomen.
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
- **Bal volgen.** `app/ball.py`: kandidaten kiezen op het spoor, inzoomen en zoeken als hij kwijt is.
- **Trainen.** `app/learning.py` (voorbeelden verzamelen, modelversies, veldnetwerk, spelerprofielen)
  en `app/train.py` (draait als apart proces). Eigen modellen staan in `data/models/custom/`.
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
