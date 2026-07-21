from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# ---- OTO YÖNLENDÝRME (Redirect) ----
@app.get("/")
def ana_sayfa():
    # Kullanýcý ana sayfaya geldiðinde direkt /docs sayfasýna þutluyoruz
    return RedirectResponse(url="/docs")

# 1. Veri Yapýsýný Tanýmlama (Pydantic Modeli)
class Urun(BaseModel):
    id: int
    isim: str
    fiyat: float
    stokta_mi: bool = True

# 2. Geçici Hafýza (Sahte Veritabaný)
urunler_db = [
    {"id": 1, "isim": "Laptop", "fiyat": 25000.0, "stokta_mi": True},
    {"id": 2, "isim": "Kablosuz Fare", "fiyat": 450.0, "stokta_mi": True}
]

# ---- [R]EAD: Tüm Ürünleri Listeleme ----
@app.get("/urunler", response_model=List[Urun])
def urunleri_listele():
    return urunler_db

# ---- [R]EAD: Tek Bir Ürünü Detaylý Getirme ----
@app.get("/urunler/{urun_id}", response_model=Urun)
def urun_detay(urun_id: int):
    for urun in urunler_db:
        if urun["id"] == urun_id:
            return urun
    # Ürün bulunamazsa 404 Hatasý fýrlatýyoruz
    raise HTTPException(status_code=404, detail="Urun bulunamadi!")

# ---- [C]REATE: Yeni Ürün Ekleme ----
@app.post("/urunler", response_model=Urun)
def urun_ekle(yeni_urun: Urun):
    # Eðer ayný ID ile ürün varsa hata verelim
    for urun in urunler_db:
        if urun["id"] == yeni_urun.id:
            raise HTTPException(status_code=400, detail="Bu ID ile bir urun zaten mevcut!")
    
    # Pydantic nesnesini standart Python sözlüðüne (dict) çevirip listeye ekliyoruz
    urunler_db.append(yeni_urun.model_dump())
    return yeni_urun

# ---- [U]PDATE: Mevcut Ürünü Güvenli Güncelleme (PUT) ----
@app.put("/urunler/{urun_id}", response_model=Urun)
def urun_guncelle(urun_id: int, guncellenen_veriler: Urun):
    hedef_index = None
    
    # 1. Aþama: Güncellenmek istenen ürün gerçekten var mý?
    for index, mevcut_urun in enumerate(urunler_db):
        if mevcut_urun["id"] == urun_id:
            hedef_index = index
            break
            
    if hedef_index is None:
        raise HTTPException(status_code=404, detail="Guncellenmek istenen urun bulunamadi!")

    # 2. Aþama: URL'deki ID ile JSON içindeki ID farklý mý? (ID deðiþtirilmek mi isteniyor?)
    if urun_id != guncellenen_veriler.id:
        # Yeni istenen ID, veritabanýndaki BAÞKA bir üründe zaten var mý?
        for urun in urunler_db:
            if urun["id"] == guncellenen_veriler.id:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Hata: {guncellenen_veriler.id} ID'li baska bir urun zaten var! ID cakismasi engellendi."
                )

    # Her þey güvenliyse güncellemeyi yapýyoruz
    yeni_veri_dict = guncellenen_veriler.model_dump()
    urunler_db[hedef_index] = yeni_veri_dict
    return yeni_veri_dict
