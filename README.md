# Match Analyzer

Een app voor je laptop die voetbalwedstrijden analyseert tot op het niveau van de individuele
speler. Je gebruikt gewone telefoonvideo's, gefilmd vanaf de zijlijn of vanaf een hoger punt en
gewoon uit de hand, en uploadt ze achteraf.

Alles draait op je eigen laptop. Er gaan geen video's of gebruiksgegevens naar internet.

## Wat het doet

| Onderdeel | Wat je krijgt |
|---|---|
| Spelers volgen | Iedereen op het veld wordt in elk beeld gevonden en over de tijd gevolgd. |
| Teams | Automatisch ingedeeld op shirtkleur, ook in zon en schaduw, en gelijk over alle video's van een wedstrijd. Scheidsrechter, keepers en publiek komen bij 'overig'. |
| Wie is wie | Rugnummers lezen (optioneel), spelers koppelen door ze in de video aan te klikken, en een koppel-assistent die de rest van het spoor van een speler voorstelt. Selecties bewaar je en neem je over in de volgende wedstrijd. |
| Per speler | Gelopen afstand, topsnelheid, sprints, minuten, heatmap en gemiddelde positie. |
| Bal | De bal volgen, balbezit per team, passes, balverlies, passnetwerk. Je kunt de bal ook zelf aanwijzen. |
| Minimap | Een 2D-bovenaanzicht dat met de video meeloopt. |
| Knippen | Een lange video vóór de analyse in delen knippen (1e/2e helft), warming-up en rust eruit. |
| Hoogtepunten | Suggesties uit de analyse (sprints, passes) en uit het geluid (gejuich, fluitsignalen). |
| Schoten en goals | Voorstellen voor schoten richting doel en mogelijke goals, uit de bal en het geluid. Bevestigd tellen ze mee: stand, tijdlijn, schotenkaart en schoten/goals per speler. |
| Clips en delen | Clips maken: begin/eind, label, spelers taggen, opmerking, tekenen op beeld (beeld bevriest), spotlight die een of meer spelers volgt, afspeellijst per speler. Export als mp4, reel of zip, en direct delen via AirDrop, WhatsApp, Berichten of Mail. |
| Zelf leren | Op jouw verzoek oefent de app op je eigen gecorrigeerde beelden, zodat hij spelers, de bal, het veld en je eigen spelers steeds beter herkent. |

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
in de app staat de versie en de commit (bijv. `versie 0.18 · 7e771cd`); die moet gelijk zijn aan
`git log -1 --oneline`. Ziet de app er na een update vreemd uit, druk dan één keer op ⌥⌘R in Safari.

Wil je ook rugnummers automatisch laten lezen? Installeer dan:

```bash
.venv/bin/pip install -r requirements-ocr.txt
```

Alle gegevens (video's, analyses, exports) staan in de map `data/`.

## Begeleide route voor een nieuwe wedstrijd

Een nieuwe wedstrijd begint in een wizard: één vraag per scherm, met bovenaan de stappen en onderaan
**Volgende**. Stop je halverwege (bijvoorbeeld omdat de analyse even duurt), dan staat er op de
wedstrijdpagina een knop **Verder met stap …**.

1. **Wedstrijd.** Naam, datum, teams en speeltijd per helft (30, 35, 40 of 45 minuten). Kies je bij
   een team een opgeslagen selectie, dan staan de spelers er meteen in.
2. **Video's.** Sleep al je video's in één keer in het vak, in willekeurige volgorde.
3. **Tijdlijn.** "Hoe laat was de aftrap?" (de app vult de opnametijd van je eerste video alvast in).
   Daarna laat de app de video zien waarvan hij denkt dat de 2e helft begint (na het langste gat
   waarin niets is gefilmd). Klopt het? Dan vul je in hoe laat de 2e helft begon; anders kies je de
   goede video. De app zet daarna alles op volgorde en rekent per video helft en minuut uit.
4. **Analyseren.** Eén keuze (Nauwkeurig of Snel) en alles gaat in de wachtrij. Je hoeft niet te
   wachten: ga gerust door.
5. **Veld vastleggen.** Per helft (je staat vaak per helft ergens anders) en per standplaats één
   video; de app stelt die met het meeste veld in beeld voor. Dat gaat in vier stapjes:
   - **a. Waar stond je?** Klik **📡 Zoek mijn plek via GPS**: de app zoekt het voetbalveld op in
     OpenStreetMap en zet je plek als 📍 op een grote veldtekening, met een gele cirkel voor de
     onnauwkeurigheid. Lukt dat niet, dan blijft de reden in beeld staan en klik je zelf op de
     tekening waar je stond.
   - **b. Plek verbeteren.** Sleep de 📍 naar waar je echt stond, kies hoe hoog je stond (staand,
     bankje, tribune) en vul de veldmaten in (of neem die van OpenStreetMap over).
   - **c. Veld intekenen.** De app zoekt zelf de witte lijnen en tekent de veldlijnen geel over het
     beeld. Kloppen ze? **✓ Ja, dit klopt**. Zo niet: probeer een ander moment of ga naar d.
   - **d. Herkenningspunten.** Klik zelf een punt aan dat je herkent (hoekvlag, middenstip, hoek van
     het strafschopgebied) en daarna hetzelfde punt op de tekening. Een balk bovenaan zegt steeds
     wat de volgende klik is; met je plek bekend zijn 1 punt en 1 lijn vaak al genoeg.
6. **Controleren.** De app kalibreert de rest van de video's zelf (zie hieronder bij *Veel video's*)
   en jij loopt de plaatjes langs: ✓ of ✗.
7. **Teams en spelers.** Klopt de kleur bij de juiste ploeg (anders omwisselen), en voer de spelers
   in of neem een vaste selectie over. Bij **✓ Klaar** koppelt de app wat hij via rugnummers kan;
   de rest doe je bij **Spelers**.

Alles wat de wizard doet, kun je daarna ook nog op de tabbladen hieronder bijstellen.

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
   - **▶ Analyseer alles** zet in één keer alle video's die nog niet geanalyseerd zijn in de
     wachtrij. De laptop doet ze één voor één (dat is sneller dan alles door elkaar); je kunt de app
     intussen gewoon gebruiken.
   - **🕒 Volgorde uit opnametijd.** Een iPhone-video weet wanneer het filmen begon. Zoals je losse
     foto's van een dag met de klok van de camera op volgorde legt, sorteert de app zo je video's.
     Een gat van 8 minuten of meer is de rust; daarna begint de 2e helft. De eerste video van een
     helft begint op de aftrap (0', of bij de 2e helft de speeltijd van één helft: 30, 40 of 45
     minuten, kies je erbij); de rest telt daar vanaf door. Je ziet eerst het voorstel en kunt
     helft en minuten aanpassen voordat je het overneemt. Begon je later met filmen dan de aftrap,
     zet dan de minuten van die helft even goed.
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
     van de **linkerpaal** of de **rechterpaal** de **onderkant** (waar hij de grond raakt) of de
     **bovenkant** (waar de lat begint, 2,44 m hoog), of een plek **op de lat**. Links en rechts zoals
     jij het doel zag vanaf je plek; de app rekent dat zelf om naar de tekening. De bovenkant en de lat hangen in de lucht en
     vertellen de app hoe ver weg het doel is en hoeveel je hebt ingezoomd. Ze tellen mee zodra de app
     weet waar je stond (zie hieronder). Na het klikken tekent de app het doel in het beeld, zodat je
     ziet of het klopt.
   - **Vertel de app waar je stond.** Klik bij **📍 Waar stond je bij het filmen?** op de veldtekening
     waar je ongeveer stond, en vul in hoe hoog je de telefoon hield. Standaard is dat 1,6 m: ooghoogte
     als je staat. Hield je hem boven je hoofd, stond je op een bankje of op de tribune, vul dan de
     echte hoogte in (of kies een snelkeuze). Rekent de app uit je klikken een andere hoogte uit, dan
     kun je die met één klik overnemen. Dan rekent de app met een
     cameramodel en is **1 punt + 1 lijn** al genoeg, bijvoorbeeld een doelpaal en de zijlijn voor je.
     Nog beter is 2 punten + 1 lijn of 1 punt + 2 lijnen; dan rekent de app ook uit of je hebt ingezoomd.
   - **Klopt een punt niet?** De blauwe kijkhoek op de tekening en de witte lijnen in het beeld volgen
     uit al je punten samen. Is er één verkeerd gekoppeld (bijvoorbeeld linker- en rechterpaal
     verwisseld), dan trekt die alles scheef. De app ziet dat aan de afwijking en zoekt de vreemde
     eend: hij laat om de beurt één punt weg en kijkt bij welk punt de rest ineens wél klopt. Dat punt
     krijgt een gele markering en er verschijnt een duidelijke waarschuwing; de kijkhoek kleurt rood.
     Verwijder het (×) of sleep het naar de goede plek.
   - **Veldmaten.** De app gaat uit van 105 × 68 m. Amateurvelden zijn vaak kleiner, bijvoorbeeld
     100 × 64. Vul de echte maten in bij **Veldmaten** (op de pagina Kalibratie); dan kloppen
     afstanden en posities beter. Het strafschopgebied en de middencirkel zijn altijd even groot.
     Na **Zoek via GPS** biedt de app aan om de maten uit OpenStreetMap over te nemen.
   - **Zoek via GPS.** iPhone-video's bevatten meestal de GPS-positie. De knop **📡 Zoek via GPS**
     zoekt het voetbalveld op in OpenStreetMap en zet je positie automatisch op de tekening. Daarvoor
     wordt alleen de coördinaat naar OpenStreetMap gestuurd, en alleen als jij op de knop klikt.
     De app vraagt alle OpenStreetMap-zoekservers tegelijk (de eerste die antwoordt wint), en
     antwoordt er geen op tijd, dan haalt hij het kaartje rond je plek rechtstreeks van
     openstreetmap.org. Een gevonden veld onthoudt de app voor die plek. Lukt het toch niet, dan zegt
     de app in gewone woorden waarom (druk, geen internet of een certificaatprobleem). De
     iPhone slaat de positie op ongeveer 5 à 10 m nauwkeurig op; klik je plek gerust preciezer aan.
   - **Veel video's? Kalibreer per standplaats maar één.** Zie het als een fotograaf op een statief:
     vanaf dezelfde plek zijn positie, hoogte en zoom gelijk, alleen de kijkrichting verschilt.
     Bovenaan de pagina Kalibratie staat **Alle video's in één keer**. De app deelt je video's in
     standplaatsen in: op GPS-positie (met de nauwkeurigheid die de iPhone erbij opslaat; precieze
     metingen tellen zwaarder) en op opnametijd. Kalibreer per standplaats één video zelf; daar volgt
     uit waar je stond, hoe hoog en hoe ver ingezoomd (ook als je je positie niet hebt ingesteld).
     Klik daarna **🤖 Kalibreer de rest automatisch**. De app legt elke andere video als een
     puzzelstuk tegen de al gekalibreerde beelden: bomen, huizen, borden en het hek staan vanaf
     dezelfde plek altijd op dezelfde plek, zoals bij een panoramafoto. Lukt dat niet (weinig
     overlap), dan zoekt hij de veldlijnen vanaf jouw plek. Een voorstel dat een camera op een
     andere plek zou opleveren, gooit de app zelf weg. Daarna loop je de plaatjes langs: liggen de
     gele lijnen op de witte? **✓** keurt goed, **✗** haalt het voorstel weg, **✎** opent de video
     om het zelf te doen. Klik op een plaatje om het groot te zien. Goedgekeurde video's helpen bij
     de volgende ronde weer mee als puzzelstuk.
   - **Laat de app het veld zoeken.** Weet de app waar je stond, dan zoekt hij zelf de witte lijnen en
     legt hij de veldtekening erop, zonder klikken (**🤖 Zoek het veld automatisch**; gebeurt vanzelf
     als er nog geen ijkmoment is). Zie het als rondkijken met een plattegrond in je hand tot alle
     lijnen kloppen. Controleer het voorstel: vallen de witte lijnen op het veld? Sleep punten bij waar
     nodig en klik **✓ Klopt**. Zegt de app **twijfel**, dan zijn er te weinig lijnen te zien (bijv.
     alleen de zijlijn); kies dan een moment met de 16-meter, middenlijn of cirkel in beeld. Klik je
     plek zo precies mogelijk aan: een paar meter ernaast maakt het zoeken een stuk lastiger. Alleen de
     GPS-positie (5 à 10 m nauwkeurig) is vaak niet genoeg.
   - **Inzoomen.** Knijp op het trackpad (of ⌥ + scrollen) om in te zoomen, bijvoorbeeld om de verre
     hoekvlag precies aan te klikken. Verschuiven doe je met twee vingers, of Shift + slepen.
     Een punt dat je ingezoomd zet, telt zwaarder mee (tot 4× bij flink inzoomen): de lijn gaat dan
     door jouw precieze stippen, en grovere punten van ver weg geven mee.
   - **Nooit gespiegeld.** Klik je alleen de middenlijn en de zijlijn aan, dan passen er twee
     oplossingen: het echte veld en zijn spiegelbeeld (vouw het veld dubbel op de middenlijn en die
     lijnen vallen op zichzelf). Een camera ziet het veld nooit in spiegelbeeld, dus de app kiest de
     echte kant. Kan dat niet, omdat de punten zelf alleen gespiegeld kloppen (bijv. het verkeerde
     doel of de verkeerde zijlijn gekozen), dan zegt de app dat; bij het ijkmoment staat dan
     **⚠ klopt niet**. Een oude kalibratie die zo gespiegeld was, zet de app vanzelf goed.
   - **Punten bewegen mee met het veld.** Zet je een punt en schuif je daarna naar een ander moment,
     dan schuift het punt mee met de camerabeweging. Zo kun je punten van verschillende momenten in één
     ijkmoment combineren: bijv. de middenstip nu en de verre hoekvlag als de camera daar is.
     Punten die daardoor buiten beeld vallen, tellen gewoon mee.
   - **Eén ijkmoment is genoeg.** (Een *ijkmoment* is een moment in de video waarop jij het veld
     hebt vastgelegd, zoals je een weegschaal één keer ijkt.) Na de analyse volgt de app de camerabeweging, zoals een lijm die
     de plattegrond op het beeld vasthoudt. Elke seconde zoekt hij daarnaast de witte veldlijnen op
     en legt hij de plattegrond er opnieuw precies op (**🤖 Automatisch bijgesteld**). Lukt dat niet,
     bijvoorbeeld bij versleten lijnen, fel zonlicht of een korrelig beeld, dan kijkt de app nog een
     keer extra goed: hij middelt over een stukje lijn in plaats van per pixel, zoals je een vage
     stoeprand beter ziet als je een paar stappen ervan bekijkt. Dit start
     vanzelf zodra je een ijkmoment opslaat, of na de analyse. De gele stippellijnen laten op elk
     moment zien hoe goed het past. Onder **🤖 Automatisch bijgesteld** staat de lijst: klik op een tijd
     om te kijken, ✓ keurt goed, ✗ verwijdert, ✎ zet de punten klaar om zelf bij te stellen, of keur
     ze met **✓ Alles accepteren** in één keer goed. Goedgekeurde blijven staan als je opnieuw bijstelt.
   - Het bijstellen werkt het best als je ook hebt ingesteld **waar je stond**: dan houdt de app je
     zoom en scheefstand vast en kan er niets ongemerkt wegglijden.
   - Kijk je lang langs alleen de zijlijn, zonder dwarslijnen zoals de 16-meterlijn, middenlijn,
     doellijn of cirkel? Dan is niet te zien hoeveel je gedraaid hebt. De app slaat die momenten over
     en overbrugt ze met de camerabeweging tot er weer dwarslijnen in beeld komen. Past het ergens
     niet, zet daar dan zelf een extra ijkmoment.

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
     ook zelf.
   - **Automatisch koppelen** koppelt tracks met een leesbaar rugnummer. De rest koppel je via de
     kaartjes of door in **Video + minimap** te pauzeren en op een speler te klikken.

   **De bal.** In **Video + minimap** zie je een cirkel om de bal: wit = gevonden, geel gestippeld =
   geschat (de bal was even niet te zien en de app trekt een lijn tussen ervoor en erna), groen = door
   jou aangewezen. Mist de app de bal op een belangrijk moment? Pauzeer, klik **⚽ Bal aanwijzen** en
   klik op de bal. Ziet de app iets anders aan voor de bal, klik dan **🚫 Geen bal hier**. Jouw
   aanwijzingen gaan altijd voor en worden gebruikt voor balbezit en passes.

4. **Schoten en goals.** Zie de app als een grensrechter bij het doel: hij let op een bal die
   ineens hard (vanaf 40 km/u) richting het doel gaat, van binnen ongeveer 30 m. Dat is een
   **schot**; gaat hij tussen de palen, dan is het **op doel**. Verdwijnt de bal daarna bij het doel
   of komt hij bij de doellijn, en hoort de app gejuich en/of een fluitsignaal, dan is het
   **waarschijnlijk een goal**. De schutter is de laatste speler die de bal had.
   - Bij **Clips & delen → Schoten en goals (voorstellen)** bevestig je met **✓ Schot** of
     **⚽ Goal**, of wijs je af met **✗**. Met **+ clip** maak je er een clip van met de schutter in de
     spotlight.
   - Mist de app er een (een hard schot is vaak even niet te zien), voeg hem dan zelf toe bij
     **Video + minimap**: pauzeer, klik de schutter aan (eventueel ook wie de assist gaf) en klik
     **🎯 Schot** of **⚽ Goal**. Met het vinkje erbij maakt de app er meteen een clip van. Weghalen
     kan met × bij "Momenten in deze video".
   - Alleen bevestigde schoten en goals tellen mee: de **stand** bovenaan de wedstrijd, de
     **tijdlijn** met de goals, de **schotenkaart** en **schoten/goals** per team en per speler bij
     **Statistieken**.

5. **Bekijken.** Bij **Statistieken** zie je de stand, de team- en spelerscijfers, heatmaps, de teamvorm,
   de schotenkaart en het passnetwerk, en kun je alles als CSV exporteren.

6. **Clips en delen.** Maak bij **Video + minimap** met één klik een clip van het moment dat je
   ziet (6 s ervoor tot 4 s erna). Pauzeer en klik eerst de spelers in het beeld aan die bij de actie
   betrokken zijn (of kies ze bij **＋ speler…**): ze komen allemaal in de clip, elk met een spotlight.
   De clip verschijnt meteen onder "Momenten in deze video" en bij **Clips & delen**. Je kunt ook bij
   **Clips & delen** een automatische suggestie omzetten in een clip: een
   sprint, een pass, of iets uit het geluid. **📣 Gejuich** betekent dat het ineens veel luider werd,
   bijvoorbeeld na een goal of grote kans; de clip begint daarom 10 s ervoor. Daarnaast vindt de app
   **🔔 Fluitsignalen**. Bij Clips & delen kun je:
   - clips verwijderen: met **×** achter een clip in de lijst, of vink er meer aan en klik
     **🗑️ Verwijder aangevinkte** (de app vraagt het eerst);
   - begin en eind per seconde verschuiven;
   - een label, spelers en een opmerking toevoegen;
   - een **spotlight** op een of meer spelers zetten: een gele ring onder de voeten met de naam, die
     de speler door de clip volgt (aan of uit per getagde speler);
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

Met telefoonbeelden vanaf de zijlijn kom je niet aan de nauwkeurigheid van een vaste 180°-camera.
Zie de cijfers als een goede indruk, niet als meetwerk.

- **Afstand en snelheid.** Hoe dichter bij de camera, hoe beter. Aan de verre kant kan een positie
  1 à 2 m afwijken.
- **Bal.** Een kleine, snelle bal is vaak even niet te zien. De app zoekt hem dan ingezoomd op en
  vult korte gaten op, maar balbezit en passes blijven een schatting. Wijs de bal zelf aan op
  momenten die ertoe doen.
- **Rugnummers.** Vanaf de zijlijn vaak onleesbaar: gebruik ze als suggestie en controleer de
  koppelingen.
- **Teams en publiek.** In fel zonlicht lijkt een donker shirt soms grijs; zo'n speler kan bij
  'overig' terechtkomen. Publiek dat langs de lijn loopt, valt pas na de kalibratie af.
- **Geluid.** Gejuich en fluitsignalen zijn suggesties; controleer ze even.
- **Schoten en goals.** De app ziet een schot alleen als hij de bal ziet wegvliegen. Een hoge bal
  lijkt verder weg dan hij is (de app rekent alsof de bal op de grond ligt). Bevestig daarom zelf, en
  voeg gemiste schoten en goals met de hand toe.
- **Spelers buiten beeld** worden niet gemeten. Het aantal minuten is de tijd dat iemand in beeld was.

## Voor ontwikkelaars

De app is gebouwd met Python (FastAPI, SQLite) en gewone HTML/JavaScript; de code staat in `app/` en
`static/`. Instellingen, zoals het detectiemodel, de analyse-fps en de sprintgrens, staan in
`app/config.py`. Tests draaien:

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```
