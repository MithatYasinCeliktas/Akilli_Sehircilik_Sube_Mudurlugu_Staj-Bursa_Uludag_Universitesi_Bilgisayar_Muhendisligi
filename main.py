import cv2
import numpy as np
import sqlite3
import math
import time
import os
from datetime import datetime
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import torch

# GPU varsa GPU (0), yoksa CPU kullan
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"[BILGI] Model calisma cihazi: {DEVICE}")

# --- 1. VERITABANI VE KLASOR KURULUMU ---
def init_db():
    if not os.path.exists("supheli_goruntuler"):
        os.makedirs("supheli_goruntuler")
    if not os.path.exists("kavga_goruntuleri"):
        os.makedirs("kavga_goruntuleri")

    conn = sqlite3.connect("sistem_kayitlari.db")
    cursor = conn.cursor()
    
    # 1. Alan Ihlalleri Tablosu
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alan_ihlalleri (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nesne_id INTEGER,
            baslangic_zamani TEXT,
            bitis_zamani TEXT,
            sure_saniye REAL
        )
    """)
    
    # 2. Supheli Canta Tablosu
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS supheli_cantalar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canta_id INTEGER,
            tarih_saat TEXT,
            resim_yolu TEXT
        )
    """)

    # 3. Kavga Kayitlari Tablosu
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS kavga_kayitlari (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tarih_saat TEXT,
            guven_skoru REAL,
            resim_yolu TEXT
        )
    """)
    conn.commit()
    conn.close()

def alan_ihlal_kaydet(nesne_id, baslangic, bitis, sure):
    conn = sqlite3.connect("sistem_kayitlari.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO alan_ihlalleri (nesne_id, baslangic_zamani, bitis_zamani, sure_saniye)
        VALUES (?, ?, ?, ?)
    """, (nesne_id, baslangic.strftime("%Y-%m-%d %H:%M:%S"), bitis.strftime("%Y-%m-%d %H:%M:%S"), round(sure, 2)))
    conn.commit()
    conn.close()
    print(f"[ALAN IHLALI LOG] ID: {nesne_id} | Sure: {round(sure, 2)} sn")

def supheli_canta_kaydet(canta_id, resim_yolu):
    conn = sqlite3.connect("sistem_kayitlari.db")
    cursor = conn.cursor()
    tarih_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO supheli_cantalar (canta_id, tarih_saat, resim_yolu)
        VALUES (?, ?, ?)
    """, (canta_id, tarih_str, resim_yolu))
    conn.commit()
    conn.close()
    print(f"[SUPHELI CANTA LOG] ID: {canta_id} | Saat: {tarih_str} | Resim: {resim_yolu}")

def kavga_kaydet(conf_score, resim_yolu):
    conn = sqlite3.connect("sistem_kayitlari.db")
    cursor = conn.cursor()
    tarih_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO kavga_kayitlari (tarih_saat, guven_skoru, resim_yolu)
        VALUES (?, ?, ?)
    """, (tarih_str, round(float(conf_score), 2), resim_yolu))
    conn.commit()
    conn.close()
    print(f"[KAVGA LOG] Saat: {tarih_str} | Guven Skoru: {round(float(conf_score), 2)} | Resim: {resim_yolu}")

def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

init_db()

# --- 2. MODELLERIN YUKLENMESI ---
model_pose = YOLO("yolov8s-pose.pt")  # Insan ve Eklem Takibi
model_bag = YOLO("yolov8s.pt")        # Canta Takibi
model_fight = YOLO("best.pt")         # Kendi Egitdiginiz Kavga Modeli

BAG_CLASSES = [24, 26, 28]  # 24: BackPack, 26: Handbag, 28: Suitcase

# --- 3. PARAMETRELER VE DEGISKENLER ---
DISTANCE_THRESHOLD = 180   # Insan-canta yakinlik siniri (piksel)
BAG_TIME_THRESHOLD = 10.0  # Sahipsiz canta alarm suresi (saniye)

bag_start_times = {}
bag_last_positions = {}
kaydedilen_cantalar = set()
aktif_alan_ihlaller = {}

# Kavga kayit cooldown kontrolu (saniyede 30 defa resim kaydetmemek icin)
son_kavga_kayit_zamani = 0
KAVGA_COOLDOWN = 5.0  # saniye

# Video Kaynagi
video_yolu = r"test.mp4"
cap = cv2.VideoCapture(video_yolu)

# Dinamik Pencere Boyutlandirma
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
MAX_W, MAX_H = 1024, 720
scale = min(MAX_W / orig_w, MAX_H / orig_h)
target_w = int(orig_w * scale)
target_h = int(orig_h * scale)

pencere_adi = "Gozetim ve Tehdit Algilama Sistemi"
cv2.namedWindow(pencere_adi, cv2.WINDOW_NORMAL)
cv2.resizeWindow(pencere_adi, target_w, target_h)

# Kirmizi Alan Poligonu
RED_ZONE_PTS = np.array([[900, 1500], [1800, 1500], [1800, 1900], [900, 1900]], np.int32)
red_polygon = Polygon(RED_ZONE_PTS)

# --- 4. ANA ISLEME DONGUSU ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    su_an = datetime.now()
    current_time_sec = time.time()

    # A) MODEL TAHMINLERI (GPU Destekli: device=0)
    results_pose = model_pose.track(frame, persist=True, classes=[0], conf=0.45, imgsz=1088, verbose=False, device=DEVICE)
    results_bag = model_bag.track(frame, persist=True, classes=BAG_CLASSES, conf=0.05, imgsz=1088, verbose=False, device=DEVICE)
    results_fight = model_fight.predict(frame, conf=0.01, imgsz=1088, verbose=False, device=DEVICE)

    person_centers = []
    mevcut_frame_ihlal_idleri = set()

    # B) KAVGA TESPIT ANALIZI
    if results_fight[0].boxes is not None and len(results_fight[0].boxes) > 0:
        f_boxes = results_fight[0].boxes.xyxy.cpu().numpy()
        f_confs = results_fight[0].boxes.conf.cpu().numpy()

        for box, conf in zip(f_boxes, f_confs):
            fx1, fy1, fx2, fy2 = map(int, box)
            
            # Turuncu/Kirmizi Kutu Cizimi
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 140, 255), 3)
            cv2.putText(frame, f"KAVGA / ARBEDE! ({round(float(conf), 2)})", (fx1, fy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
            
            # Ekranın Üstüne Genel Alarm Yazısı
            cv2.putText(frame, "!!! KAVGA / ARBEDE TESPIT EDILDI !!!", (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            # Fotoğraf alma ve log kaydi (5 saniyede bir defa)
            if (current_time_sec - son_kavga_kayit_zamani) > KAVGA_COOLDOWN:
                zaman_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = f"kavga_goruntuleri/kavga_{zaman_str}.jpg"
                cv2.imwrite(img_path, frame)
                kavga_kaydet(conf, img_path)
                son_kavga_kayit_zamani = current_time_sec

    # C) INSAN TESPITI VE KIRMIZI ALAN IHLAL ANALIZI
    if results_pose[0].boxes is not None and results_pose[0].boxes.id is not None and results_pose[0].keypoints is not None:
        p_boxes = results_pose[0].boxes.xyxy.cpu().numpy()
        p_track_ids = results_pose[0].boxes.id.cpu().numpy().astype(int)
        keypoints_data = results_pose[0].keypoints.data.cpu().numpy()

        for box, track_id, kpts in zip(p_boxes, p_track_ids, keypoints_data):
            x1, y1, x2, y2 = map(int, box)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            person_centers.append((center_x, center_y))

            # Eklem Noktalari
            l_hip, r_hip = kpts[11], kpts[12]
            l_knee, r_knee = kpts[13], kpts[14]
            l_ankle, r_ankle = kpts[15], kpts[16]

            # Zemin/Ayak Hesabi
            if l_ankle[2] > 0.3 or r_ankle[2] > 0.3:
                foot_x = center_x
                foot_y = int(max(l_ankle[1], r_ankle[1]))
            elif (l_knee[2] > 0.3 or r_knee[2] > 0.3) and (l_hip[2] > 0.3 or r_hip[2] > 0.3):
                hip_y = (l_hip[1] + r_hip[1]) / 2
                knee_y = (l_knee[1] + r_knee[1]) / 2
                uyluk_boyu = abs(knee_y - hip_y)
                foot_x = center_x
                foot_y = int(knee_y + (uyluk_boyu * 1.1))
            else:
                foot_x = center_x
                foot_y = y2

            # Alan Kontrolu
            foot_point = Point(foot_x, foot_y)
            is_inside = red_polygon.contains(foot_point)

            if is_inside:
                mevcut_frame_ihlal_idleri.add(track_id)
                box_color = (0, 0, 255) # Kirmizi

                if track_id not in aktif_alan_ihlaller:
                    aktif_alan_ihlaller[track_id] = {"baslangic": su_an, "son_gorulme": su_an}
                else:
                    aktif_alan_ihlaller[track_id]["son_gorulme"] = su_an

                gecen_sure = (su_an - aktif_alan_ihlaller[track_id]["baslangic"]).total_seconds()
                p_label = f"ALAN IHLALI! ID:{track_id} ({round(gecen_sure, 1)}s)"
            else:
                box_color = (0, 255, 0) # Yesil
                p_label = f"ID:{track_id}"

            # Cizimler (Insan)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(frame, (foot_x, foot_y), 5, (255, 0, 255), -1)
            cv2.putText(frame, p_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # D) CANTA TESPITI VE SAHIPSIZ CANTA ANALIZI
    if results_bag[0].boxes is not None and results_bag[0].boxes.id is not None:
        b_boxes = results_bag[0].boxes.xyxy.cpu().numpy()
        b_track_ids = results_bag[0].boxes.id.cpu().numpy().astype(int)

        for box, bag_id in zip(b_boxes, b_track_ids):
            bx1, by1, bx2, by2 = map(int, box)
            bag_center = (int((bx1 + bx2) / 2), int((by1 + by2) / 2))

            min_dist = float('inf')
            for p_center in person_centers:
                d = calculate_distance(bag_center, p_center)
                if d < min_dist:
                    min_dist = d

            is_stationary = True
            if bag_id in bag_last_positions:
                prev_pos = bag_last_positions[bag_id]
                if calculate_distance(bag_center, prev_pos) > 15:
                    is_stationary = False

            bag_last_positions[bag_id] = bag_center
            bag_color = (255, 0, 0) # MAVI

            if min_dist > DISTANCE_THRESHOLD and is_stationary:
                if bag_id not in bag_start_times:
                    bag_start_times[bag_id] = current_time_sec
                
                elapsed = current_time_sec - bag_start_times[bag_id]

                if elapsed >= BAG_TIME_THRESHOLD:
                    bag_color = (0, 0, 255) # KIRMIZI
                    bag_status = f"ALARM: SUPHELI CANTA! ({int(elapsed)}s)"
                    
                    cv2.putText(frame, "!!! SUPHELI CANTA TESPIT EDILDI !!!", (40, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                    if bag_id not in kaydedilen_cantalar:
                        zaman_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        img_path = f"supheli_goruntuler/canta_ID{bag_id}_{zaman_str}.jpg"
                        cv2.imwrite(img_path, frame)
                        supheli_canta_kaydet(bag_id, img_path)
                        kaydedilen_cantalar.add(bag_id)
                else:
                    bag_status = f"Canta ID:{bag_id} (Sahipsiz: {int(elapsed)}s)"
            else:
                bag_start_times.pop(bag_id, None)
                kaydedilen_cantalar.discard(bag_id)
                bag_status = f"Canta ID:{bag_id} (Guvenli)"

            # Cizimler (Canta)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), bag_color, 2)
            cv2.putText(frame, bag_status, (bx1, by1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bag_color, 2)

    # E) ALANDAN CIKAN INSANLARIN KAYDI
    tamamlanan_ids = []
    for track_id, bilgi in aktif_alan_ihlaller.items():
        if track_id not in mevcut_frame_ihlal_idleri:
            if (su_an - bilgi["son_gorulme"]).total_seconds() > 1.0:
                toplam_sure = (bilgi["son_gorulme"] - bilgi["baslangic"]).total_seconds()
                if toplam_sure > 0.5:
                    alan_ihlal_kaydet(track_id, bilgi["baslangic"], bilgi["son_gorulme"], toplam_sure)
                tamamlanan_ids.append(track_id)

    for tid in tamamlanan_ids:
        del aktif_alan_ihlaller[tid]

    # F) EKRAN CIZIMLERI (Kirmizi Yasakli Alan)
    cv2.polylines(frame, [RED_ZONE_PTS], isClosed=True, color=(0, 0, 255), thickness=3)
    cv2.putText(frame, "YASAKLI ALAN", (RED_ZONE_PTS[0][0], RED_ZONE_PTS[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow(pencere_adi, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 5. TEMIZLIK VE SIFIRLAMA ---
print("\n[BILGI] Video sonlandirildi. Aktif alan ihlalleri kaydediliyor...")
for track_id, bilgi in aktif_alan_ihlaller.items():
    toplam_sure = (bilgi["son_gorulme"] - bilgi["baslangic"]).total_seconds()
    if toplam_sure > 0.5:
        alan_ihlal_kaydet(track_id, bilgi["baslangic"], bilgi["son_gorulme"], toplam_sure)

cap.release()
cv2.destroyAllWindows()
