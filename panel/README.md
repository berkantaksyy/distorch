# panel

Tek dosyalık test paneli. **Servis değil** — Tkinter penceresi açılır.
**Linux/V4L2**, macOS'a özgü hiçbir şey yok. CPU yeterli.

Hiçbir şeyi import etmiyor, hiçbir dosyana dokunmuyor. Distorsiyon matematiği
(Brown k1,k2 + çapa dönüşümü) panelin içinde kopya olarak duruyor.

## Kurulum

```bash
cd panel
./run.sh
```

Bu kadar. `run.sh` kurulum eksikse `setup.sh`'i kendi çağırır, sonra paneli
açar. Sanal ortamı elle aktive etmene gerek yok.

Kurulumu ayrı yapmak istersen:

```bash
./setup.sh            # kur
./setup.sh --core     # sadece çekirdek: torch/YOLO yok, ~1 dk
./setup.sh --diag     # sadece tanı, hiçbir şey kurmaz
./setup.sh --force    # sanal ortamı sıfırdan
```

### Kurulum katmanlı — "her şey ya da hiçbir şey" değil

| katman | ne kurar | patlarsa |
|---|---|---|
| sistem | `python3-venv python3-tk v4l-utils libgl1 libglib2.0-0` | komutu ekrana yazar, bir kez elle çalıştırırsın |
| çekirdek | numpy, opencv-headless, pillow (`requirements.txt`) | panel açılmaz — asıl sorun budur |
| ağır | torch, torchvision, ultralytics, scipy (`requirements-extra.txt`) | **panel yine açılır**, sadece `.pt` model / YOLO bölümü çalışmaz |

torch mimariye göre seçiliyor: x86_64'te resmi CPU deposundan (PyPI tekerleği
CUDA taşıyor, ~2.5 GB ve bu makinede hiçbir işe yaramıyor), aarch64'te PyPI'dan
(CPU deposunda aarch64 tekerleği yok). Biri olmazsa diğerine düşüyor.

Kurulumun sonunda `panel.py --check` çalışır ve panelin açılıp açılmayacağını
söyler. Bunu tek başına da çağırabilirsin:

```bash
.venv/bin/python panel.py --check
```

```
== ortam
  python      3.12.3   /home/aco/panel/.venv/bin/python
  sistem      Linux 6.8.0  x86_64
  numpy       2.1.3
  opencv      4.11.0
  tkinter     8.6
== istege bagli paketler  (yoksa panel yine acilir)
  var  torch          2.6.0+cpu  (distorch .pt modeli)
  ...
== matematik
  OK  distort/undistort geri donus 4.55e-13 px, harita cikti
== kamera
  v4l2 backend  var
  /dev/video0
panel acilabilir.
```

## Sorun giderme

| ne görüyorsun | ne yap |
|---|---|
| `tkinter hala yok` | `sudo apt install -y python3-tk` sonra `./setup.sh --force` |
| `pencere acilamadi` / `no display name` | Panelin masaüstü oturumu lazım. SSH'tan açacaksan `ssh -X kullanici@makine` |
| `acilamadi: /dev/video0 (v4l2)` | `ls -l /dev/video*` — aygıt var mı; `sudo usermod -aG video "$USER"` sonra **oturumu kapat/aç** |
| `kare okunamadi` | Kamera o format/çözünürlüğü vermiyor: `v4l2-ctl -d /dev/video0 --list-formats-ext` |
| `Qt platform plugin "xcb"` | GUI'li opencv bulaşmış. `./setup.sh` bunu kendi temizliyor, tekrar çalıştır |
| `torch kurulamadi` | Panel yine açılır. Sadece `.pt` modeli çalışmaz, profil JSON ile devam edebilirsin. Ayrıntı: `/tmp/panel_torch.log` |

Loglar: `/tmp/panel_sys.log`, `panel_venv.log`, `panel_core.log`,
`panel_torch.log`, `panel_extra.log`.

## Bölümler

**Kamera** — aygıt, çözünürlük, fps, format. Varsayılan **1920×1080, YUYV
(YUY2), 30 fps, V4L2**. Kamera istediğini veremezse durum çubuğu ne aldığını
yazar. `tara` düğmesi `/dev/video*`'u yeniden listeler.

> **İki ayrı model var, karıştırma.** İkisi de `.pt` ama alakaları yok:
> **distorch ağırlığı** (distort_v2.pt, distort_v3.pt) lens bozulmasını çözer —
> çıktısı k1, k2, cx, cy. **YOLO ağırlığı** (Plastic.pt, Glass.pt, Metal_All.pt)
> şişeyi bulur — çıktısı kutu/maske. Panel dosyayı seçerken hangisi olduğunu
> arşivin içinden anlıyor; yanlış yuvaya koyarsan açık bir uyarıyla reddediyor.

**1) Distorsiyon düzeltme** — üç mod:

- `kapalı` — ham kare
- `profil JSON` — distorch'un yazdığı profil (`camera_matrix` + `dist_coeffs`),
  veya düz `{"k1":..,"k2":..,"cx":..,"cy":..}`. İçinde `roll_deg` varsa okunur.
- `distorch modeli (.pt)` — **istediğin ağırlık**: distort_v2, v3, ne olursa. Yanındaki
  `*_meta.json` otomatik aranır (normalizasyon + çapalar oradan gelir),
  `*_bias.json` varsa sapma düzeltmesi uygulanır.

θ ilk karede bir kez çözülür, sonra sabit kalır — boş hazneyi bir kez kalibre
edip sonra şişeyle test edebilmen için. Modeli/profili değiştirince sıfırlanır.

`roll'u sıfırla` — girdiğin açı kadar kareyi düzleştirir. **Aynı remap'in
içinde**, yani kare bir kez yeniden örnekleniyor, ek maliyet yok. Profilde
`roll_deg` varsa kutuya kendi gelir.

İlk düzeltilmiş karede durum çubuğunda `haritalar hazirlaniyor...` yazar:
1920×1080 için remap haritası 1–2 saniye sürer, sonra önbellekten gelir.

**Kesme** — dört kaydırıcı: üst / alt / sol / sağ, **yüzde** olarak.
Düzeltmeden **sonra** uygulanır. Varsayılan 0, yani hiç kesmez. Ne kadar
keseceğine sen karar ver. `sıfırla` hepsini 0 yapar.

**2) Nesne tespiti (YOLO)** — model `.pt` seç, `çalıştır`ı işaretle. Segmentasyon modeli ise
maske konturu (mavi), `minAreaRect` (yeşil) ve eksen-hizalı kutu (kırmızı)
birlikte çizilir; etikette **en/boy** oranı ve açı yazar. `mm/px` girersen
mm cinsinden ölçü de eklenir. `retina_masks` maskeyi proto ızgarası yerine
girdi çözünürlüğünde hesaplatır.

Önizlemede YOLO her karede değil, **bir önceki koşu ne kadar sürdüyse o kadar
bekleyip** tekrar çalışır; aradaki karelerde son tespit yeniden çizilir. CPU'da
bir koşu 1–2 saniye sürüyor, her karede çalıştırılsa arayüz hiç nefes alamazdı.
`KARE AL`'da böyle bir bekleme yok: kaydedilen her kare kendi tespitiyle kaydedilir.

**Kayıt** — etiket, adet (varsayılan **3**), kareler arası ms.
`KARE AL` her basışta ayrı bir klasör açar:

```
cikti/test_20260917_141203/
  01_ham.png            # kameradan geldiği gibi
  01_duzeltilmis.png    # düzeltme + kesme sonrası
  01_tespit.jpg         # YOLO çizimli (tespit varsa)
  02_... 03_...
  kayit.json            # theta, kesme ayarları, kamera, her tespitin en/boy ve mm ölçüsü
```

`kayit.json` ayarları da içerdiği için hangi kareyi hangi ayarla aldığını
sonradan karıştırmazsın. Varsayılan çıktı klasörü `panel.py`'nin yanındaki
`cikti/` — nereden çalıştırdığın fark etmez.

## Komut satırı

```bash
./run.sh --device /dev/video2 --width 1920 --height 1080 \
         --fourcc YUYV --fps 30 --shots 3 --gap-ms 150 --out ./cikti

./run.sh --check          # pencere açmadan ortamı sına
```

## Notlar

- `opencv-python-headless` kullanılıyor: panel Tkinter olduğu için OpenCV'nin
  GUI'sine gerek yok, Qt/GTK bağımlılığı gelmiyor. ultralytics bağımlılık olarak
  GUI'li sürümü çekerse `setup.sh` onu kaldırıp headless'ı geri koyuyor.
- `YOLO_AUTOINSTALL=false` ayarlanıyor: ultralytics çalışma anında kendi kendine
  pip install denemesin. Paket kurmak `setup.sh`'in işi.
- YUYV 1920×1080'de birçok USB kamera 5–10 fps verir (bant genişliği). Akıcı
  önizleme istersen `MJPG` seç — ama ölçüm karelerini YUYV'de al, MJPG sıkıştırma
  kenarları bozar.
- θ ilk karede çözülüyor. Hazneyi değiştirdiysen modu bir kapatıp açman yeter.
