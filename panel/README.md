# panel

Tek dosyalık test paneli. **Servis değil** — Tkinter penceresi açılır.
**Linux/V4L2**, macOS'a özgü hiçbir şey yok. CPU yeterli.

Hiçbir şeyi import etmiyor, hiçbir dosyana dokunmuyor. Distorsiyon matematiği
(Brown k1,k2 + çapa dönüşümü) panelin içinde kopya olarak duruyor.

## Kurulum

```bash
cd panel
./setup.sh
source .venv/bin/activate
python panel.py
```

`setup.sh` tkinter'ı kontrol eder, venv kurar, paketleri yükler, `/dev/video*`
aygıtlarını ve destekledikleri formatları listeler.

Kamera açılmazsa kullanıcı `video` grubunda olmayabilir:

```bash
sudo usermod -aG video "$USER"    # sonra oturumu kapat/aç
```

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

**Kesme** — dört kaydırıcı: üst / alt / sol / sağ, **yüzde** olarak.
Düzeltmeden **sonra** uygulanır. Varsayılan 0, yani hiç kesmez. Ne kadar
keseceğine sen karar ver. `sıfırla` hepsini 0 yapar.

**2) Nesne tespiti (YOLO)** — model `.pt` seç, `çalıştır`ı işaretle. Segmentasyon modeli ise
maske konturu (mavi), `minAreaRect` (yeşil) ve eksen-hizalı kutu (kırmızı)
birlikte çizilir; etikette **en/boy** oranı ve açı yazar. `mm/px` girersen
mm cinsinden ölçü de eklenir. `retina_masks` maskeyi proto ızgarası yerine
girdi çözünürlüğünde hesaplatır.

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
sonradan karıştırmazsın.

## Komut satırı

```bash
python panel.py --device /dev/video2 --width 1920 --height 1080 \
                --fourcc YUYV --fps 30 --shots 3 --gap-ms 150 --out ./cikti
```

## Notlar

- `opencv-python-headless` kullanılıyor: panel Tkinter olduğu için OpenCV'nin
  GUI'sine gerek yok, Qt/GTK bağımlılığı gelmiyor.
- YUYV 1920×1080'de birçok USB kamera 5–10 fps verir (bant genişliği). Akıcı
  önizleme istersen `MJPG` seç — ama ölçüm karelerini YUYV'de al, MJPG sıkıştırma
  kenarları bozar.
- θ ilk karede çözülüyor. Hazneyi değiştirdiysen modu bir kapatıp açman yeter.
