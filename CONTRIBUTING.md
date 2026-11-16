# Za ekipo: kako začeti in kako delamo

Oddaja ekipe HatTrio za *More than a Hack 2027*, 1. faza: za vsako sličico tekme Barcelona–Athletic
napovemo `H` (domači imajo žogo), `A` (gostje), `D` (mrtva žoga) ali `X` (zunaj polčasov).
Rok za oddajo: **17. 11. 2026 ob 23:59 CET**.

Najprej preberi:

- [README.md](README.md): namestitev, ukazi za oddajo in za učenje od začetka;
- [docs/technical_document.md](docs/technical_document.md): kaj model dela, kaj je bilo preizkušeno
  in zakaj je (ali ni) sprejeto, z vsemi številkami;
- [third_party/mtah2027/](third_party/mtah2027/): pravila tekmovanja (`RULES.md`, `FAQ.md`) in
  ocenjevanje (`EVALUATION.md`).

## 1. Namestitev

Potrebuješ Python 3.12 in [uv](https://docs.astral.sh/uv/); na macOS še `brew install libomp`
(za LightGBM).

```bash
git clone https://github.com/zk4621/hattrio-possession.git
```

```bash
cd hattrio-possession && uv sync
```

## 2. Podatki tekme

Podatki **niso** v repozitoriju in vanj **ne smejo** priti (pravila tekmovanja). Paket organizatorjev
si preneseš sam in ga razpakiraš v `data/` (mapa je v `.gitignore`):

```
data/
  tactical/BAR-ATH_tactical_atd.xml, _atd.txt, _homography.csv, _smart_tagging.xml
  broadcast/BAR-ATH_broadcast_atd.xml, _atd.txt, _homography.csv
```

ali nastaviš `MTAH_DATA_DIR` na mapo, kamor si ga razpakiral.

## 3. Preveri, da vse deluje

```bash
uv run python scripts/make_submission.py --team HatTrio --proba model
```

Izpisati mora SHA-256 `fdaf2ed296c5f9ab21eaab832e6f57e17034b481facdf53d2885154a164d1a9e` (to je
oddana datoteka `outputs/HatTrio.csv`). Nato še testi:

```bash
uv run pytest -q
```

## 4. Za učenje in nove poskuse

- DFL tekme (2,6 GB, v `external/dfl/`): `uv run python scripts/fetch_external.py`
- Metrica vzorčne tekme: `git clone https://github.com/metrica-sports/sample-data external/metrica`,
  nato `git -C external/metrica checkout e706dd5` (različica, na kateri so naučeni oddani modeli)
- Mreža čez zaporedje potrebuje PyTorch (`uv run --with torch==2.15.0 ...`, glej README).
  PyTorch in LightGBM nikoli ne učimo v istem procesu (dva OpenMP runtima se sesujeta), zato imajo
  skripte ločena koraka `train` in `evaluate`.

Napovedi trenutne verzije, s katerimi primerjaš svojo idejo (vsaka tekma napovedana z modelom, ki je
ni videl):

```bash
uv run python scripts/check_metrica_relabel.py lgb <dir D>                  # LightGBM, ~20 min
uv run --with torch==2.15.0 python scripts/test_sequence_model.py train <dir E>  # mreža, ~11 min
```

Celoten recept, s katerim iz repozitorija znova nastanejo vsi štirje oddani modeli (preverjeno: bajt
za bajtom enaki), je v README, razdelek *Reproduce the submitted models*.

Dober vzorec za nov poskus je [scripts/test_self_training.py](scripts/test_self_training.py): vzame
`--oof <dir D> --nn <dir E>`, zgradi kandidata in ga primerja z osnovo na enak način.

## 5. Pravila tekmovanja, ki jih ne smemo prekršiti

- **Brez ročnih oznak na tej tekmi**, v nobeni količini in za noben namen. Oznake mora narediti
  algoritem. Oznake, ki jih ustvari model sam (psevdo-oznake), so dovoljene.
- **Javnih 5 minut resnice** (`third_party/mtah2027/evaluation/gt/`) uporabljamo samo za preverjanje,
  da cevovod deluje. Nikoli za učenje, nastavljanje ali izbiro med verzijami.
- **Podatkov tekme ne nalagaj v oblačna orodja z umetno inteligenco** (ChatGPT, Claude ipd.) in jih ne
  objavljaj nikjer. Kodo pokažeš lahko; vsebine `data/` ali `cache/` pa ne, tudi ne izsekov.
  Skripte, ki berejo podatke tekme, izpisujejo samo skupne številke (deleže, štetja, ocene).
- **Repozitorij mora natanko ponoviti oddano datoteko.**

## 6. Kako delamo

1. Vsak dela na svoji veji in odpre pull request na `main`. Na `main` ne pošiljaj neposredno.
2. **Pravilo določi pred meritvijo** in ga zapiši na vrh skripte (z datumom), npr. »sprejmemo, če
   je boljše na vseh treh Metrica tekmah in povprečje zraste vsaj za 0,005, DFL povprečje pa ne pade
   pod …«. Po meritvi pravila ne spreminjamo.
3. **Ocenjujemo na treh Metrica tekmah** (isti ponudnik sledenja kot naša tekma), vsaka napovedana z
   modelom, ki je ni videl, dekoder izbran na drugih dveh. DFL tekme so preverjanje, da se nič ne
   pokvari (njihove oznake se zamenjajo ob vsakem dotiku žoge, zato niso glavno merilo).
4. Mera je **spellF1** (0–1): posesti se ujemajo z resničnimi po času prekrivanja. Natančnost po
   sličicah ni pomembna sama po sebi; razbita posest je draga.
5. Rezultat zapiši v tehnični dokument (razdelek 3), tudi če ni sprejet, in poročilo s skupnimi
   številkami v `models/`.
6. Če se na `main` spremenijo modeli: na novo ustvari `outputs/HatTrio.csv`, posodobi SHA-256 v
   README in tehničnem dokumentu (razdelek 5) in preveri, da `uv run pytest -q` uspe.
7. Nikoli ne commitaj `data/`, `cache/`, `outputs/` ali `external/` (vse je v `.gitignore`).

## 7. Kje je največ prostora za izboljšave

Na Metrica tekmah trenutna verzija doseže spellF1 0,777. Pri prejšnji verziji (0,730, pred popravkom
Metrica oznak) bi popolna izbira ekipe na sličicah v igri dvignila oceno na 0,88, popolna ločitev
žive in mrtve žoge pa le na 0,77. Največ se torej izgubi pri **trenutku predaje** in pri **kratkih
posestih**.
Kaj je že bilo preizkušeno (in zakaj ni pomagalo), je v tehničnem dokumentu, razdelek 3.
