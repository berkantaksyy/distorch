# distorch — delik destekli otomatik lens kalibrasyonu

Bos hazne karesinden, o cihaza ozel distorsiyon profilini **tek komutla** cikarir.
Insan mudahalesi yok, hedef tahtasi yok, kamera ayarina bagimlilik yok.

```bash
bash setup.sh
python -m distorch.calibrate kare.jpg --device ACO_6600_0003 --overlay
```

```
ACO_ANKA_0045   1920x1080
cnn        k1 -1.3521 k2 +1.9002 cx 995.4 cy 604.2   (baslangic + bekci)
net        k1 -1.3527 k2 +2.0732 cx 995.4 cy 604.2   (start + guard)
stage1     holes 7  spread 1.09%  step 249.0 px = 80 mm -> 0.3213 mm/px
edges      top 152 bottom 77 left 14 right 14   rms 0.477 px
stage2     rings 5 (seeds 1)  ratio 1.9365  USED step 476 px
theta      k1 -1.4125  k2 +2.4175  cx 987.1  cy 617.6
quality    corner 0.86 px (stage1 4.86)  residual 1.06  symmetry 2.5
guard      net differs by 6.3 px (threshold 14.0)
verdict    ACCEPT
profile    measurements/ACO_ANKA_0045.json      time 1.44 s
```

## Neden calisiyor

Panelin alt sirasindaki **7 delik, merkezden merkeze gercek 80 mm**. Karede
boyutu bilinen tek nesne bu. Duzluk (plumb-line) tek basina olcegi ve dagilim
merkezini cozemez; esit aralik ikisini de cozer.

| | duzeltmesiz | duzeltilmis |
|---|---|---|
| ayni sisenin kadraj boyunca olculen boyu | 172–243 px (%41 fark) | 247–250 px (%1) |

## Akis

```
kare
 ├─ net.py      distort_v3 -> theta          (baslangic degeri + BEKCI)
 ├─ holes.py    7 delik, alt-piksel elips merkezleri
 ├─ edges.py    panel kenarlari, alt-piksel
 ├─ solve.py    ASAMA 1: kenar + delik         -> theta1
 ├─ rings.py    ASAMA 2: bilezik (tahmin + yerel olcum)
 │              solve  kenar + delik + bilezik  -> theta2
 │              KAPI: kose hatasini dusurmezse theta2 ATILIR
 ├─ quality.py  kapilar -> accept / warn / reject
 └─ profile.py  JSON
```

**Cikti her zaman geometrik cozumdur.** CNN karar vermez; baslangic degeri verir
ve sonucu denetler. Torch kurulu degilse sistem calismaya devam eder, yalnizca
bekci kapisi devre disi kalir.

## Olculen sonuclar (106 saha karesi, 51 cihaz)

| olcut | medyan | p90 | en kotu | hedef |
|---|---|---|---|---|
| kose hatasi | 2.50 px | 3.39 | 4.40 | < 8 medyan |
| delik yayilimi | %1.05 | %1.87 | %2.73 | < %1.5 medyan, < %3 max |
| cizgi RMS | 0.72 px | 1.39 | 2.13 | < 2.0 ort |
| mm/px | 0.3226 ± 0.0019 (%0.6) | | | |
| kalibrasyon suresi | 0.20 s/kare | | | < 2 s |

Karar dagilimi: **93 accept · 13 warn · 0 reject**.
Bilezikler 49/106 karede kullanildi.

### Ust serit
Kadrajin ustundeki parlak serit (y~100-320, panelden genis) fite 0.2 agirlikla
giriyor. Panelin ustunde baska olcum yok; serit olmadan fit orayi ekstrapole
ediyor ve serit kenari duzeltmeden sonra gozle gorulur kavisli kaliyor.

| | serit yok | serit var |
|---|---|---|
| kose hatasi (medyan) | 4.02 px | **2.50 px** |
| kose hatasi (en kotu) | 8.74 px | **4.40 px** |
| serit kenar artigi | 1.13 px | **0.97 px** |
| panel genisligi, yukseklik boyunca | %0.36 | **%0.31** |
| delik yayilimi | %0.93 | %1.05 |

Agirlik 0.2 olcumle secildi: serit ~420 nokta getiriyor ve tam agirlikta fiti
delik sirasindan koparıyor (yayilim %1.37'ye cikiyor). 0.2'de kose kazancinin
tamami zaten var.

**Egitimde hic gorulmemis cihazlarda** (ACO_6600_0003 / 0015, ACO_ANKA_0045),
delik araligi yayilimi:

| yontem | yayilim |
|---|---|
| tek profil filoya uygulanirsa | %5.63 |
| distort_v3 tek basina | %5.12 |
| **distorch** | **%0.32** |

## Dosyalar

| | |
|---|---|
| `distorch/geom.py` | distorsiyon matematigi (f=1920 sabit, Brown k1/k2) |
| `distorch/net.py` | distort_v3 cikarimi + sistematik sapma duzeltmesi |
| `distorch/holes.py` | aydinlik panel delik dedektoru |
| `distorch/edges.py` | panel kenari alt-piksel cikarimi |
| `distorch/rings.py` | bilezikli panel: tahmin + yerel olcum |
| `distorch/solve.py` | geometrik cozucu (cizgi + esit aralik kisitlari) |
| `distorch/quality.py` | kose hatasi, kapilar |
| `distorch/profile.py` | profil JSON yaz/oku |
| `distorch/calibrate.py` | TEK GIRIS |
| `tools/batch.py` | tum kareleri kalibre et, filo raporu |
| `tools/compare.py` | model vs sistem: tek metrik + gorsel |
| `tools/split_undistorted.py` | zaten duzeltilmis kareleri ayir |
| `tests/` | 26 test, 5 dosya |

## Testler

```bash
for t in geom holes solve net pipeline; do python -m tests.test_$t; done
```

## Model ile karsilastirma

```bash
python -m tools.compare                 # tum veri seti, tek metrik
python -m tools.compare ACO_ANKA_0045   # o kare icin tam gorsel
```

**Metrik: olcum tutarsizligi (%)** — panelin 7 deligi gercekte esit arailkli
(80 mm). Duzeltilmis goruntude komsu araliklar olculur, (max-min)/ort. Yani
"ayni cisim kadrajin farkli yerlerinde olculdugunde boyu yuzde kac degisir".

| grup | n | duzeltmesiz | model | sistem | kazanc |
|---|---|---|---|---|---|
| TUM VERI SETI | 106 | %34.52 | %5.74 | **%0.95** | 6.1x |
| TEST (model gormedi) | 8 | %33.95 | %5.11 | %0.86 | 5.9x |
| VAL | 8 | %34.86 | %4.87 | %0.50 | 9.7x |
| TRAIN (model gordu) | 90 | %34.49 | %5.90 | %1.07 | 5.5x |

Sistemin en kotu karesi (%2.77) modelin en iyi karesinden (%2.91) iyi; 106
karenin 106'sinda sistem modelden iyi.

## Ogrenilen tuzaklar

- **Ters distorsiyonda sabit nokta yetmez.** k1 ≈ −1.35'te kosede 7 px hata
  birakiyor; Newton ile bitirilmeli (`geom.radial_inverse`).
- **Kenar cikariminda maske DOLU olmali.** Delikleri doldurulmamis maskeyle
  sutun taramasi panelin degil deligin sinirini yakalar: alt kenar 99 noktadan
  1'e duser.
- **Kapa (anchor) tanimi:** `m = radial_inverse(a)/a`, yani duzeltme buyutmesi.
  `m = g(a)` DEGIL — o tanim k1'in isaretini ters cevirir.
- **Iki noktali sira fite hicbir sey katmaz.** n noktali sira 2n−4 bilgi tasir.
- **CNN ile geometri arasindaki fark sistematiktir**, rastgele degil
  (v3 plinebig'e nisan alarak egitildi). Sapma cikarilmadan kapi kurulamaz.
- **Kapi delik saymaz.** Onemli olan sayi degil yayilim: ortadaki delik kaybi
  kose hatasini 5.2 -> 7.0 px yapar, iki uc delik kaybi 124 px.
- **Leave-one-out calismaz.** 7 noktadan birini cikarmak cozumu zaten
  zayiflatiyor; temiz karede bile 43 px. Yerine nokta artigi + yayilim.
- **Sabit esik calismaz.** Panel medyani filoda 199–247 arasinda geziyor.
- **k3 eklenemez.** Ucuncu radyal terim iki kez denendi (serit fitte varken ve
  yokken). Filo genelinde tutarsiz: k3 medyan -2.04 ama std 2.55, isaret 7
  karede pozitif 21 karede negatif, bir karede cx 684 px kaydi. Gurultuye
  oturuyor, lense degil.
