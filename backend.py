"""
Kiazi Smart — AI Backend (FastAPI)
-----------------------------------
Backend hii kamili (auth, mashamba, mizunguko ya mimea, shughuli,
matumizi, AI disease-scans, na soko) inayotumiwa moja kwa moja na
kiazi-smart.html yako. Kila kitu kinahifadhiwa kwenye faili la SQLite
(kiazi.db) folder moja na backend hii, hivyo data (akaunti, mashamba,
uchunguzi wa AI, soko) inabaki hata baada ya kuzima na kuwasha seva
tena, na inaonekana kwenye vifaa vyote vinavyounganishwa na seva hii.

HAIBADILISHI chochote kwenye HTML/UI yako — ni server tu inayojibu
maombi yale yale ambayo HTML tayari imeandikwa kuyatuma (angalia
function `api()` na `routeApi()` kwenye HTML: hizo ndizo njia
(endpoints) ambazo backend hii inatekeleza upande wa seva).

JINSI YA KUENDESHA (VS Code):
1. Weka faili hizi TATU folder moja:
     - backend.py (hili)
     - potato_disease_mobilenetv2_best.keras
     - kiazi-smart.html   (HTML yako halisi, bila kubadilishwa)
2. pip install -r requirements_backend.txt
3. Endesha:  uvicorn backend:app --reload --port 8000
4. Fungua kivinjari: http://localhost:8000/
   -> Hii inaonyesha INTERFACE YAKO HALISI (kiazi-smart.html) kama
      ilivyokuwa, bila kubadilika kabisa.
5. Ndani ya app hiyo, nenda "Mipangilio > Anwani ya Seva ya AI" na
   weka:  http://localhost:8000
   Baada ya hapo akaunti, mashamba, AI Scan na Soko zitatumia seva
   hii halisi (backendMode='real') badala ya modi ya ndani (local).

KUMBUKA: token za kuingia (session) zinahifadhiwa kwenye kumbukumbu
(memory) ya mchakato wa seva pekee (kama TOKENS ilivyokuwa kwenye
HTML) — kuzima/kuwasha seva (au --reload kuanzisha upya mchakato)
kutahitaji kuingia (login) tena, lakini akaunti, mashamba na data
nyingine zote zinabaki salama kwenye kiazi.db.
"""

import io
import json
import secrets
import hashlib
import hmac
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image

# ============================ MIPANGILIO ============================

MODEL_PATH = Path(__file__).parent / "potato_disease_mobilenetv2_best.keras"
HTML_PATH = Path(__file__).parent / "kiazi-smart.html"
DB_PATH = Path(__file__).parent / "kiazi.db"
IMG_SIZE = (224, 224)

# Mpangilio huu unalingana na Dense(4, softmax) ya model yako, kufuata
# mpangilio wa kawaida wa alfabeti wa Keras wakati wa training.
CLASS_NAMES = [
    "Not_Potato_Leaf",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]

ROLE_PERMISSIONS = {
    "farmer": ["dashboard", "farms", "scan", "market", "profile"],
    "buyer": ["dashboard", "market", "profile"],
    "admin": ["dashboard", "farms", "scan", "market", "profile", "admin"],
}
ALLOWED_ROLES = {"farmer", "buyer"}

# NOTE: token za kuingia (sessions) zinahifadhiwa kwenye jedwali la
# "sessions" ndani ya kiazi.db (siyo tena kwenye kumbukumbu ya
# mchakato/memory pekee). Hii ni muhimu hasa unapoendesha na
# `--reload`: uvicorn hufuatilia faili zote za folder hii (ikiwemo
# kiazi.db), na kila andiko jipya kwenye DB (mfano baada ya login)
# linaweza kusababisha --reload kuanzisha upya mchakato mzima wa
# server. Token ya kumbukumbu tu ingefutwa papo hapo (ndiyo chanzo cha
# ujumbe "muda wa kuingia umeisha" mara tu baada ya kuingia). Kwa
# kuhifadhi kwenye DB, token inabaki sahihi hata seva ikianzishwa upya.

app = FastAPI(title="Kiazi Smart AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================ HELPERS ============================


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def to_num(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 130_000)
    return f"{salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, hexhash = stored.split("$", 1)
    except (ValueError, AttributeError):
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 130_000)
    return hmac.compare_digest(dk.hex(), hexhash)


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                phone_number TEXT,
                role TEXT NOT NULL,
                location_region TEXT,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                demo INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS farms (
                id TEXT PRIMARY KEY,
                owner_email TEXT NOT NULL,
                name TEXT NOT NULL,
                region TEXT,
                size_acres REAL,
                soil_type TEXT,
                latitude REAL,
                longitude REAL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS crop_cycles (
                id TEXT PRIMARY KEY,
                farm_id TEXT NOT NULL,
                variety TEXT,
                planting_date TEXT,
                expected_harvest_date TEXT,
                stage TEXT,
                is_active INTEGER,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS activities (
                id TEXT PRIMARY KEY,
                farm_id TEXT NOT NULL,
                crop_cycle_id TEXT,
                activity_type TEXT,
                description TEXT,
                activity_date TEXT,
                photo TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS expenses (
                id TEXT PRIMARY KEY,
                farm_id TEXT NOT NULL,
                category TEXT NOT NULL,
                amount REAL NOT NULL,
                currency TEXT,
                expense_date TEXT,
                notes TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS disease_scans (
                id TEXT PRIMARY KEY,
                farm_id TEXT NOT NULL,
                crop_cycle_id TEXT,
                predicted_class TEXT,
                confidence REAL,
                all_probabilities TEXT,
                source TEXT,
                model_note TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS market_listings (
                id TEXT PRIMARY KEY,
                posted_by TEXT,
                seller_name TEXT NOT NULL,
                phone TEXT NOT NULL,
                region TEXT NOT NULL,
                product TEXT,
                quantity_kg REAL NOT NULL,
                price_per_kg REAL NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL
            );
            """
        )


def ensure_demo_accounts():
    demos = [
        ("Admin wa Demo", "admin@kiazi.tz", "Admin123!", "admin"),
        ("Mkulima wa Demo", "farmer@kiazi.tz", "Farmer123!", "farmer"),
        ("Mnunuzi wa Viazi wa Demo", "buyer@kiazi.tz", "Buyer123!", "buyer"),
    ]
    with db() as conn:
        for name, email, pw, role in demos:
            existing = conn.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone()
            if not existing:
                conn.execute(
                    """INSERT INTO users
                       (id, full_name, email, phone_number, role, location_region,
                        password_hash, created_at, demo)
                       VALUES (?,?,?,?,?,?,?,?,1)""",
                    (new_id("usr"), name, email, None, role, "Mbeya",
                     hash_password(pw), now_iso()),
                )


def user_to_public(row: sqlite3.Row) -> dict:
    d = dict(row)
    d.pop("password_hash", None)
    d["demo"] = bool(d.get("demo"))
    d["permissions"] = ROLE_PERMISSIONS.get(d["role"], ROLE_PERMISSIONS["farmer"])
    return d


def get_email(request: Request, required: bool = True) -> Optional[str]:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    token = None
    if auth and auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    email = None
    if token:
        with db() as conn:
            row = conn.execute("SELECT email FROM sessions WHERE token=?", (token,)).fetchone()
            if row:
                email = row["email"]
    if required and not email:
        raise HTTPException(status_code=401, detail="Muda wa kuingia umeisha. Tafadhali ingia tena.")
    return email


def get_owned_farm(conn: sqlite3.Connection, farm_id: str, email: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM farms WHERE id=?", (farm_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Shamba halipo au limeshafutwa.")
    if row["owner_email"] != email:
        raise HTTPException(status_code=403, detail="Huna ruhusa ya kufikia shamba hili.")
    return row


def require_admin(conn: sqlite3.Connection, email: str):
    row = conn.execute("SELECT role FROM users WHERE email=?", (email,)).fetchone()
    if not row or row["role"] != "admin":
        raise HTTPException(status_code=403, detail="Huna ruhusa ya msimamizi.")


def run_prediction(raw_bytes: bytes) -> dict:
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Imeshindwa kusoma picha. Jaribu picha nyingine.")

    model = get_model()
    img_resized = image.resize(IMG_SIZE)
    arr = np.array(img_resized, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    all_probabilities = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    predicted_class = max(all_probabilities, key=all_probabilities.get)
    confidence = all_probabilities[predicted_class]
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": all_probabilities,
    }


_model = None


def get_model():
    global _model
    if _model is None:
        # Model ina Lambda layer inayoita "preprocess_input" (jina tu).
        # custom_objects inaambia Keras function hiyo hasa inatoka wapi
        # (mobilenet_v2.preprocess_input) - bila hii utapata:
        # "Could not locate function 'preprocess_input'".
        _model = tf.keras.models.load_model(
            MODEL_PATH,
            safe_mode=False,
            custom_objects={
                "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input
            },
        )
    return _model


@app.on_event("startup")
def warm_up():
    init_db()
    ensure_demo_accounts()
    # Pakia model mara moja app inapoanza, isije ikapakiwa kwa mara ya
    # kwanza wakati mtumiaji anasubiri jibu la scan ya kwanza.
    get_model()


# ============================ UKURASA / STATIC ============================


@app.get("/")
def home():
    if not HTML_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Sikuona HTML kwenye {HTML_PATH}. Weka faili lako la interface hapo.",
        )
    return FileResponse(HTML_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "service": "kiazi-smart-ai-backend"}


@app.get("/manifest.json")
def manifest():
    # HTML inarejea ./manifest.json (PWA). Hii inatoa manifest ndogo halali
    # ili isirudishe 404 kwenye logs — haiathiri UI yako kwa vyovyote.
    return JSONResponse(
        {
            "name": "Kiazi Smart",
            "short_name": "Kiazi Smart",
            "start_url": "/",
            "display": "standalone",
            "background_color": "#ffffff",
            "theme_color": "#16a34a",
            "icons": [],
        }
    )


@app.get("/sw.js")
def service_worker():
    # HTML inajaribu kusajili ./sw.js (na ina .catch() ikishindwa), lakini
    # tunatoa faili tupu halali ili isionekane kama 404 kwenye logs.
    return Response(content="// no-op service worker\n", media_type="application/javascript")


# ============================ AUTH ============================


@app.post("/api/auth/register")
def register(payload: dict = Body(...)):
    email = str(payload.get("email") or "").strip().lower()
    full_name = str(payload.get("full_name") or "").strip()
    password = payload.get("password") or ""
    if not email or not full_name or not password:
        raise HTTPException(status_code=400, detail="Tafadhali jaza jina, barua pepe na nenosiri.")
    if len(str(password)) < 6:
        raise HTTPException(status_code=400, detail="Nenosiri liwe na angalau herufi 6.")
    role = payload.get("role") if payload.get("role") in ALLOWED_ROLES else "farmer"

    with db() as conn:
        existing = conn.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="Barua pepe hii tayari imesajiliwa. Jaribu kuingia.")
        conn.execute(
            """INSERT INTO users
               (id, full_name, email, phone_number, role, location_region,
                password_hash, created_at, demo)
               VALUES (?,?,?,?,?,?,?,?,0)""",
            (
                new_id("usr"), full_name, email,
                payload.get("phone_number"), role, payload.get("location_region"),
                hash_password(str(password)), now_iso(),
            ),
        )
    return {"ok": True}


@app.post("/api/auth/login")
def login(payload: dict = Body(...)):
    email = str(payload.get("email") or "").strip().lower()
    password = str(payload.get("password") or "")
    with db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not row or not verify_password(password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Barua pepe au nenosiri si sahihi.")
    token = secrets.token_urlsafe(24)
    with db() as conn:
        conn.execute(
            "INSERT INTO sessions (token, email, created_at) VALUES (?,?,?)",
            (token, email, now_iso()),
        )
    return {"access_token": token, "user": user_to_public(row)}


# ============================ ADMIN ============================


@app.get("/api/admin/users")
def admin_list_users(request: Request):
    email = get_email(request)
    with db() as conn:
        require_admin(conn, email)
        rows = conn.execute("SELECT * FROM users ORDER BY created_at").fetchall()
        return [user_to_public(r) for r in rows]


@app.post("/api/admin/reset-password")
def admin_reset_password(request: Request, payload: dict = Body(...)):
    email = get_email(request)
    new_password = str(payload.get("new_password") or "")
    target_email = str(payload.get("email") or "").strip().lower()
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="Nenosiri jipya liwe na angalau herufi 6.")
    with db() as conn:
        require_admin(conn, email)
        row = conn.execute("SELECT 1 FROM users WHERE email=?", (target_email,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Mtumiaji hakupatikana.")
        conn.execute("UPDATE users SET password_hash=? WHERE email=?", (hash_password(new_password), target_email))
    return {"ok": True}


# ============================ FARMS ============================


@app.get("/api/farms")
def list_farms(request: Request):
    email = get_email(request)
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM farms WHERE owner_email=? ORDER BY created_at DESC", (email,)
        ).fetchall()
        return [dict(r) for r in rows]


@app.post("/api/farms")
def create_farm(request: Request, payload: dict = Body(...)):
    email = get_email(request)
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Jina la shamba linahitajika.")
    farm = {
        "id": new_id("farm"),
        "owner_email": email,
        "name": name,
        "region": payload.get("region"),
        "size_acres": to_num(payload.get("size_acres")),
        "soil_type": payload.get("soil_type"),
        "latitude": to_num(payload.get("latitude")),
        "longitude": to_num(payload.get("longitude")),
        "created_at": now_iso(),
    }
    with db() as conn:
        conn.execute(
            """INSERT INTO farms
               (id, owner_email, name, region, size_acres, soil_type, latitude, longitude, created_at)
               VALUES (:id,:owner_email,:name,:region,:size_acres,:soil_type,:latitude,:longitude,:created_at)""",
            farm,
        )
    return farm


@app.delete("/api/farms/{farm_id}")
def delete_farm(farm_id: str, request: Request):
    email = get_email(request)
    with db() as conn:
        get_owned_farm(conn, farm_id, email)
        conn.execute("DELETE FROM farms WHERE id=?", (farm_id,))
        conn.execute("DELETE FROM crop_cycles WHERE farm_id=?", (farm_id,))
        conn.execute("DELETE FROM activities WHERE farm_id=?", (farm_id,))
        conn.execute("DELETE FROM expenses WHERE farm_id=?", (farm_id,))
        conn.execute("DELETE FROM disease_scans WHERE farm_id=?", (farm_id,))
    return {"ok": True}


# ============================ CROP CYCLES ============================


@app.get("/api/farms/{farm_id}/crop-cycles")
def list_cycles(farm_id: str, request: Request):
    email = get_email(request)
    with db() as conn:
        get_owned_farm(conn, farm_id, email)
        rows = conn.execute(
            "SELECT * FROM crop_cycles WHERE farm_id=? ORDER BY created_at DESC", (farm_id,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["is_active"] = bool(d["is_active"])
        out.append(d)
    return out


@app.post("/api/farms/{farm_id}/crop-cycles")
def create_cycle(farm_id: str, request: Request, payload: dict = Body(...)):
    email = get_email(request)
    stage = payload.get("stage") or "planted"
    cycle = {
        "id": new_id("cyc"),
        "farm_id": farm_id,
        "variety": payload.get("variety"),
        "planting_date": payload.get("planting_date"),
        "expected_harvest_date": payload.get("expected_harvest_date"),
        "stage": stage,
        "is_active": 0 if stage == "harvested" else 1,
        "created_at": now_iso(),
    }
    with db() as conn:
        get_owned_farm(conn, farm_id, email)
        conn.execute(
            """INSERT INTO crop_cycles
               (id, farm_id, variety, planting_date, expected_harvest_date, stage, is_active, created_at)
               VALUES (:id,:farm_id,:variety,:planting_date,:expected_harvest_date,:stage,:is_active,:created_at)""",
            cycle,
        )
    cycle["is_active"] = bool(cycle["is_active"])
    return cycle


# ============================ ACTIVITIES ============================


@app.get("/api/farms/{farm_id}/activities")
def list_activities(farm_id: str, request: Request):
    email = get_email(request)
    with db() as conn:
        get_owned_farm(conn, farm_id, email)
        rows = conn.execute(
            "SELECT * FROM activities WHERE farm_id=? ORDER BY created_at DESC", (farm_id,)
        ).fetchall()
        return [dict(r) for r in rows]


@app.post("/api/farms/{farm_id}/activities")
def create_activity(farm_id: str, request: Request, payload: dict = Body(...)):
    email = get_email(request)
    activity = {
        "id": new_id("act"),
        "farm_id": farm_id,
        "crop_cycle_id": payload.get("crop_cycle_id"),
        "activity_type": payload.get("activity_type") or "other",
        "description": payload.get("description"),
        "activity_date": payload.get("activity_date") or now_iso()[:10],
        "photo": payload.get("photo"),
        "created_at": now_iso(),
    }
    with db() as conn:
        get_owned_farm(conn, farm_id, email)
        conn.execute(
            """INSERT INTO activities
               (id, farm_id, crop_cycle_id, activity_type, description, activity_date, photo, created_at)
               VALUES (:id,:farm_id,:crop_cycle_id,:activity_type,:description,:activity_date,:photo,:created_at)""",
            activity,
        )
    return activity


@app.delete("/api/farms/{farm_id}/activities/{activity_id}")
def delete_activity(farm_id: str, activity_id: str, request: Request):
    email = get_email(request)
    with db() as conn:
        get_owned_farm(conn, farm_id, email)
        conn.execute("DELETE FROM activities WHERE farm_id=? AND id=?", (farm_id, activity_id))
    return {"ok": True}


# ============================ EXPENSES ============================


@app.get("/api/farms/{farm_id}/expenses/summary")
def expense_summary(farm_id: str, request: Request):
    email = get_email(request)
    with db() as conn:
        get_owned_farm(conn, farm_id, email)
        rows = conn.execute(
            "SELECT category, amount FROM expenses WHERE farm_id=?", (farm_id,)
        ).fetchall()
    total = 0.0
    by_category: dict[str, float] = {}
    for r in rows:
        amt = r["amount"] or 0
        total += amt
        by_category[r["category"]] = by_category.get(r["category"], 0) + amt
    return {"total_expenses": total, "by_category": by_category}


@app.get("/api/farms/{farm_id}/expenses")
def list_expenses(farm_id: str, request: Request):
    email = get_email(request)
    with db() as conn:
        get_owned_farm(conn, farm_id, email)
        rows = conn.execute(
            "SELECT * FROM expenses WHERE farm_id=? ORDER BY created_at DESC", (farm_id,)
        ).fetchall()
        return [dict(r) for r in rows]


@app.post("/api/farms/{farm_id}/expenses")
def create_expense(farm_id: str, request: Request, payload: dict = Body(...)):
    email = get_email(request)
    category = payload.get("category")
    amount_val = to_num(payload.get("amount"))
    if not category or amount_val is None or amount_val < 0:
        raise HTTPException(status_code=400, detail="Jaza aina na kiasi cha gharama kwa usahihi.")
    expense = {
        "id": new_id("exp"),
        "farm_id": farm_id,
        "category": category,
        "amount": amount_val,
        "currency": payload.get("currency") or "TZS",
        "expense_date": payload.get("expense_date") or now_iso()[:10],
        "notes": payload.get("notes"),
        "created_at": now_iso(),
    }
    with db() as conn:
        get_owned_farm(conn, farm_id, email)
        conn.execute(
            """INSERT INTO expenses
               (id, farm_id, category, amount, currency, expense_date, notes, created_at)
               VALUES (:id,:farm_id,:category,:amount,:currency,:expense_date,:notes,:created_at)""",
            expense,
        )
    return expense


@app.delete("/api/farms/{farm_id}/expenses/{expense_id}")
def delete_expense(farm_id: str, expense_id: str, request: Request):
    email = get_email(request)
    with db() as conn:
        get_owned_farm(conn, farm_id, email)
        conn.execute("DELETE FROM expenses WHERE farm_id=? AND id=?", (farm_id, expense_id))
    return {"ok": True}


# ============================ DISEASE SCANS (AI) ============================


@app.get("/api/farms/{farm_id}/disease-scans")
def list_scans(farm_id: str, request: Request, crop_cycle_id: Optional[str] = None):
    email = get_email(request)
    with db() as conn:
        get_owned_farm(conn, farm_id, email)
        if crop_cycle_id:
            rows = conn.execute(
                "SELECT * FROM disease_scans WHERE farm_id=? AND crop_cycle_id=? ORDER BY created_at DESC",
                (farm_id, crop_cycle_id),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM disease_scans WHERE farm_id=? ORDER BY created_at DESC", (farm_id,)
            ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["all_probabilities"] = json.loads(d["all_probabilities"]) if d["all_probabilities"] else {}
        out.append(d)
    return out


@app.post("/api/farms/{farm_id}/disease-scans")
async def create_scan(
    farm_id: str,
    request: Request,
    file: UploadFile = File(...),
    crop_cycle_id: Optional[str] = None,
):
    email = get_email(request)
    with db() as conn:
        get_owned_farm(conn, farm_id, email)

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Faili lililopakiwa siyo picha.")
    raw = await file.read()
    result = run_prediction(raw)

    record = {
        "id": new_id("scan"),
        "farm_id": farm_id,
        "crop_cycle_id": crop_cycle_id,
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "all_probabilities": json.dumps(result["all_probabilities"]),
        "source": "ai_model",
        "model_note": None,
        "created_at": now_iso(),
    }
    with db() as conn:
        conn.execute(
            """INSERT INTO disease_scans
               (id, farm_id, crop_cycle_id, predicted_class, confidence, all_probabilities,
                source, model_note, created_at)
               VALUES (:id,:farm_id,:crop_cycle_id,:predicted_class,:confidence,:all_probabilities,
                       :source,:model_note,:created_at)""",
            record,
        )
    record["all_probabilities"] = result["all_probabilities"]
    return record


# ============================ MARKET (SOKO) ============================


@app.get("/api/market")
def list_market(region: Optional[str] = None):
    with db() as conn:
        if region and region != "Zote":
            rows = conn.execute(
                "SELECT * FROM market_listings WHERE region=? ORDER BY created_at DESC", (region,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM market_listings ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]


@app.post("/api/market")
def create_listing(request: Request, payload: dict = Body(...)):
    email = get_email(request, required=False)
    seller_name = str(payload.get("seller_name") or "").strip()
    phone = str(payload.get("phone") or "").strip()
    region = payload.get("region")
    quantity_kg = to_num(payload.get("quantity_kg")) or 0
    price_per_kg = to_num(payload.get("price_per_kg")) or 0
    if not seller_name or not phone or not region or not (quantity_kg > 0) or not (price_per_kg > 0):
        raise HTTPException(
            status_code=400,
            detail="Jaza jina, namba ya simu, mkoa, kiasi (kg) na bei kwa usahihi.",
        )
    listing = {
        "id": new_id("lst"),
        "posted_by": email or "anonymous",
        "seller_name": seller_name,
        "phone": phone,
        "region": region,
        "product": payload.get("product") or "Viazi Mviringo",
        "quantity_kg": quantity_kg,
        "price_per_kg": price_per_kg,
        "notes": payload.get("notes"),
        "created_at": now_iso(),
    }
    with db() as conn:
        conn.execute(
            """INSERT INTO market_listings
               (id, posted_by, seller_name, phone, region, product, quantity_kg, price_per_kg, notes, created_at)
               VALUES (:id,:posted_by,:seller_name,:phone,:region,:product,:quantity_kg,:price_per_kg,:notes,:created_at)""",
            listing,
        )
    return listing


@app.delete("/api/market/{listing_id}")
def delete_listing(listing_id: str, request: Request):
    get_email(request)
    with db() as conn:
        conn.execute("DELETE FROM market_listings WHERE id=?", (listing_id,))
    return {"ok": True}


# ============================ AI PREDICT (moja kwa moja, bila kuhifadhi) ============================


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Faili lililopakiwa siyo picha.")
    raw = await file.read()
    return run_prediction(raw)
