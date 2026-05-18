"""
台股情報站 backend - FastAPI + yfinance + FinMind + Telegram
啟動: python server.py  →  http://localhost:18505/

功能:
- 即時價格 / OHLC / 技術指標 (yfinance)
- 三大法人 (FinMind 免費端點)
- 持久化觀察清單 (watchlist.json)
- 訊號偵測 (黃金/死亡交叉、KD 超買賣、突破 20 日新高低)
- 多週期 K 線 (日/週/月)
- 新聞 (yfinance 內建)
- Telegram 警示 (每 5 分鐘背景掃描)
- 熱度榜 / 族群分組
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
WATCHLIST_FILE = ROOT / "watchlist.json"
ALERTS_FILE         = ROOT / "alerts.json"
ALERTS_LOG_FILE     = ROOT / "alerts_log.json"
TELEGRAM_FILE       = ROOT / "telegram.json"
PORTFOLIO_FILE      = ROOT / "portfolio.json"
REBALANCE_TARGET_FILE = ROOT / "rebalance_target.json"
GROUP_ALERTS_FILE   = ROOT / "group_alerts.json"
GEMINI_FILE    = ROOT / "gemini.json"

# ----------------------------------------------------------------------------
# Default watchlist (首次啟動時寫入 watchlist.json)
# ----------------------------------------------------------------------------
DEFAULT_WATCHLIST: dict[str, dict[str, str]] = {
    # 1. 晶圓代工 (Foundry)
    "2330": {"name": "台積電",   "tag": "晶圓代工 · 先進製程",       "yf": "2330.TW",  "group": "晶圓代工"},

    # 2. IC 設計 (含 IP)
    "2454": {"name": "聯發科",   "tag": "IC 設計 · 手機 SoC",        "yf": "2454.TW",  "group": "IC 設計"},
    "3661": {"name": "世芯-KY",  "tag": "IP / ASIC 設計服務",        "yf": "3661.TW",  "group": "IP / 矽智財"},

    # 3. 探針卡 / 封測 (含 Image 2 帶出的封測領導股)
    "3711": {"name": "日月光投控","tag": "封測 · 全球龍頭",            "yf": "3711.TW",  "group": "封測"},
    "2449": {"name": "京元電",   "tag": "封測 · 測試",               "yf": "2449.TW",  "group": "封測"},
    "6515": {"name": "穎崴",     "tag": "封測 · 測試介面",            "yf": "6515.TW",  "group": "封測"},
    "6223": {"name": "旺矽",     "tag": "封測 · 探針卡",             "yf": "6223.TWO", "group": "封測"},
    "6510": {"name": "精測",     "tag": "封測 · 探針卡",             "yf": "6510.TWO", "group": "封測"},
    "6217": {"name": "中探針",   "tag": "封測 · 探針卡",             "yf": "6217.TWO", "group": "封測"},
    "6147": {"name": "頎邦",     "tag": "封測 · 驅動 IC",            "yf": "6147.TWO", "group": "封測"},

    # 4. 先進封裝設備 (CoWoS)
    "6187": {"name": "萬潤",     "tag": "先進封裝設備 (CoWoS)",       "yf": "6187.TWO", "group": "先進封裝"},
    "3131": {"name": "弘塑",     "tag": "先進封裝設備 · 清洗",       "yf": "3131.TWO", "group": "先進封裝"},
    "7734": {"name": "印能科技", "tag": "半導體 · 烘烤製程設備",     "yf": "7734.TWO", "group": "先進封裝"},
    "8027": {"name": "鈦昇",     "tag": "IC 封裝設備",               "yf": "8027.TWO", "group": "先進封裝"},

    # 5. 矽晶圓 / 上游材料
    "3532": {"name": "台勝科",   "tag": "矽晶圓 · 磊晶",             "yf": "3532.TW",  "group": "矽晶圓"},
    "6488": {"name": "環球晶",   "tag": "矽晶圓 · 全球前段",         "yf": "6488.TWO", "group": "矽晶圓"},

    # 6. ABF 載板 (三劍客)
    "8046": {"name": "南電",     "tag": "ABF 載板",                   "yf": "8046.TW",  "group": "ABF 載板"},
    "3189": {"name": "景碩",     "tag": "ABF 載板",                   "yf": "3189.TW",  "group": "ABF 載板"},
    "3037": {"name": "欣興",     "tag": "ABF 載板",                   "yf": "3037.TW",  "group": "ABF 載板"},

    # 7. 高速 CCL / PCB
    "2383": {"name": "台光電",   "tag": "高速銅箔基板 (CCL)",         "yf": "2383.TW",  "group": "高速 CCL"},
    "4958": {"name": "臻鼎-KY",  "tag": "PCB / 軟板",                 "yf": "4958.TW",  "group": "PCB / 軟板"},

    # 8. 矽光子 / CPO 光通訊
    "6442": {"name": "光聖",     "tag": "光通訊 · 主動光纜",          "yf": "6442.TW",  "group": "矽光子 / CPO"},
    "4979": {"name": "華星光",   "tag": "光通訊 · 光收發",            "yf": "4979.TWO", "group": "矽光子 / CPO"},
    "6451": {"name": "訊芯-KY",  "tag": "光通訊 · CPO 模組",          "yf": "6451.TW",  "group": "矽光子 / CPO"},
    "3163": {"name": "波若威",   "tag": "光通訊 · 矽光子",            "yf": "3163.TWO", "group": "矽光子 / CPO"},
    "3081": {"name": "聯亞",     "tag": "光通訊 · 雷射晶粒",          "yf": "3081.TWO", "group": "矽光子 / CPO"},

    # 9. 伺服器代工 (ODM)
    "2317": {"name": "鴻海",     "tag": "伺服器代工 · ODM",           "yf": "2317.TW",  "group": "伺服器 ODM"},
    "6669": {"name": "緯穎",     "tag": "AI 伺服器代工",              "yf": "6669.TW",  "group": "伺服器 ODM"},

    # 10. 散熱模組
    "3017": {"name": "奇鋐",     "tag": "散熱模組 · 液冷",            "yf": "3017.TW",  "group": "散熱模組"},
    "3324": {"name": "雙鴻",     "tag": "散熱模組",                   "yf": "3324.TW",  "group": "散熱模組"},

    # 11. 電源管理 / BBU 備援電池
    "2308": {"name": "台達電",   "tag": "電源管理 · AI 電源",         "yf": "2308.TW",  "group": "電源 / BBU"},
    "6121": {"name": "新普",     "tag": "BBU 備援電池",               "yf": "6121.TWO", "group": "電源 / BBU"},

    # 12. BMC 伺服器管理晶片
    "5274": {"name": "信驊",     "tag": "BMC 伺服器管理晶片",         "yf": "5274.TW",  "group": "BMC / 周邊"},

    # === AI 價值鏈往上層 (應用/模型/能源) ===
    # 13. 機器人 / AI 應用 (蛋糕第 5 層)
    "2395": {"name": "研華",     "tag": "工業電腦 · 邊緣 AI",         "yf": "2395.TW",  "group": "機器人 / AI 應用"},
    "8210": {"name": "所羅門",   "tag": "AI 機器視覺 / 自動化",       "yf": "8210.TW",  "group": "機器人 / AI 應用"},
    "4938": {"name": "和碩",     "tag": "代工 · 機器人 / AI 終端",    "yf": "4938.TW",  "group": "機器人 / AI 應用"},

    # 14. AI 模型 / 主權 AI (蛋糕第 4 層)
    "6770": {"name": "力積電",   "tag": "晶圓代工 · 主權 AI 合作",    "yf": "6770.TW",  "group": "AI 模型 / 主權 AI"},

    # 15. 重電 / 電網 (蛋糕第 1 層 — AI 電力基礎)
    "1519": {"name": "華城",     "tag": "重電 · 大型變壓器",          "yf": "1519.TW",  "group": "重電 / 電網"},
    "1513": {"name": "中興電",   "tag": "重電 · 配電盤 / 開關",       "yf": "1513.TW",  "group": "重電 / 電網"},
    "1503": {"name": "士電",     "tag": "重電 · 變壓器 / 配電",       "yf": "1503.TW",  "group": "重電 / 電網"},
}

CACHE_TTL = 300
_cache: dict[str, tuple[float, Any]] = {}
_lock = threading.Lock()


def cache_get(key: str):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    return None


def cache_set(key: str, val: Any) -> None:
    _cache[key] = (time.time(), val)


def cache_set_ttl(key: str, val: Any, ttl_seconds: int) -> None:
    """自訂 TTL 寫入快取 (覆蓋預設 CACHE_TTL)。用於 Gemini 等高成本資料。"""
    _cache[key] = (time.time() + ttl_seconds - CACHE_TTL, val)


# ----------------------------------------------------------------------------
# 持久化
# ----------------------------------------------------------------------------
def load_json(p: Path, default: Any) -> Any:
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def save_json(p: Path, data: Any) -> None:
    with _lock:
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _migrate_watchlist_groups(wl: dict) -> dict:
    """套用 DEFAULT_WATCHLIST 的最新族群分類到既有清單。
    規則：
    - 既有代號若在 DEFAULT 內，更新 group / tag / name 為新版（族群細分）
    - 既有代號不在 DEFAULT 內（user 自行加的）→ 保留不動
    - DEFAULT 有但既有清單沒有 → 不自動加入（避免 user 刻意刪除的又跑回來）
    """
    changed = False
    for code, default_meta in DEFAULT_WATCHLIST.items():
        if code in wl:
            cur = wl[code]
            # 只有 group 或 tag 不同才更新 (yf 不動)
            if cur.get("group") != default_meta["group"] or cur.get("tag") != default_meta["tag"]:
                cur["group"] = default_meta["group"]
                cur["tag"]   = default_meta["tag"]
                if not cur.get("name"):
                    cur["name"] = default_meta["name"]
                changed = True
    if changed:
        save_json(WATCHLIST_FILE, wl)
        print(f"[migrate] watchlist 族群分類已套用最新版")
    return wl


def load_watchlist() -> dict:
    wl = load_json(WATCHLIST_FILE, None)
    if wl is None:
        save_json(WATCHLIST_FILE, DEFAULT_WATCHLIST)
        return DEFAULT_WATCHLIST.copy()
    return _migrate_watchlist_groups(wl)


def load_alerts() -> dict:
    return load_json(ALERTS_FILE, {})


def load_telegram() -> dict:
    return load_json(TELEGRAM_FILE, {"bot_token": "", "chat_id": ""})


# ----------------------------------------------------------------------------
# 技術指標
# ----------------------------------------------------------------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def rsi_indicator(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd_indicator(s: pd.Series):
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    return dif, dea, (dif - dea) * 2


def kd_indicator(df: pd.DataFrame, n: int = 9):
    low_n = df["Low"].rolling(n).min()
    high_n = df["High"].rolling(n).max()
    rsv = (df["Close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    return k, d


def safe(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        if np.isnan(v) or np.isinf(v):
            return None
        return float(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return v


# ----------------------------------------------------------------------------
# 訊號偵測
# ----------------------------------------------------------------------------
def detect_signals(closes: pd.Series, ma5_s: pd.Series, ma20_s: pd.Series,
                   k_s: pd.Series, d_s: pd.Series, rsi_s: pd.Series,
                   highs: pd.Series, lows: pd.Series, period: str = "D") -> list[dict]:
    sigs: list[dict] = []
    # 各週期的 breakout 視窗與標籤
    breakout_cfg = {
        "D": (20, "20 日"),
        "W": (20, "20 週"),
        "M": (12, "12 月"),
    }
    bk_n, bk_label = breakout_cfg.get(period, breakout_cfg["D"])

    if len(closes) < max(21, bk_n + 1):
        return sigs

    # 均線黃金 / 死亡交叉 (MA5 vs MA20)
    if not pd.isna(ma5_s.iloc[-1]) and not pd.isna(ma20_s.iloc[-2]):
        if ma5_s.iloc[-2] <= ma20_s.iloc[-2] and ma5_s.iloc[-1] > ma20_s.iloc[-1]:
            sigs.append({"key": "golden_cross", "label": "🌟黃金交叉", "color": "red"})
        elif ma5_s.iloc[-2] >= ma20_s.iloc[-2] and ma5_s.iloc[-1] < ma20_s.iloc[-1]:
            sigs.append({"key": "death_cross", "label": "💀死亡交叉", "color": "green"})

    # KD 黃金 / 死亡交叉
    if len(k_s) >= 2:
        last_k, prev_k = float(k_s.iloc[-1]), float(k_s.iloc[-2])
        last_d, prev_d = float(d_s.iloc[-1]), float(d_s.iloc[-2])
        if prev_k <= prev_d and last_k > last_d and last_k < 50:
            sigs.append({"key": "kd_cross_up", "label": "🔼KD 黃金交叉", "color": "red"})
        elif prev_k >= prev_d and last_k < last_d and last_k > 50:
            sigs.append({"key": "kd_cross_dn", "label": "🔽KD 死亡交叉", "color": "green"})
        if last_k > 80:
            sigs.append({"key": "kd_overbought", "label": "⚠️KD 超買", "color": "orange"})
        elif last_k < 20:
            sigs.append({"key": "kd_oversold", "label": "🚀KD 超賣", "color": "cyan"})

    # RSI 超買 / 超賣
    if not pd.isna(rsi_s.iloc[-1]):
        rsi_v = float(rsi_s.iloc[-1])
        if rsi_v > 75:
            sigs.append({"key": "rsi_overbought", "label": "🔥RSI 超買", "color": "orange"})
        elif rsi_v < 25:
            sigs.append({"key": "rsi_oversold", "label": "❄️RSI 超賣", "color": "cyan"})

    # 突破新高 / 跌破新低
    price = float(closes.iloc[-1])
    high_n = float(highs.iloc[-(bk_n+1):-1].max())
    low_n  = float(lows.iloc[-(bk_n+1):-1].min())
    if price > high_n * 1.001:
        sigs.append({"key": "breakout_high", "label": f"🚀 突破 {bk_label}新高", "color": "red"})
    if price < low_n * 0.999:
        sigs.append({"key": "breakdown_low", "label": f"📉 跌破 {bk_label}新低", "color": "green"})

    return sigs


# ----------------------------------------------------------------------------
# 三大法人 (FinMind)
# ----------------------------------------------------------------------------
def fetch_institutional(stock_id: str, days: int = 10) -> dict | None:
    cached = cache_get(f"inst:{stock_id}")
    if cached:
        return cached

    try:
        end = pd.Timestamp.today()
        start = end - pd.Timedelta(days=40)
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
            "data_id": stock_id,
            "start_date": start.strftime("%Y-%m-%d"),
        }
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return None

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["net"] = (df["buy"] - df["sell"]) / 1000

        FI_NAMES = {"Foreign_Investor", "Foreign_Dealer_Self", "外資自營商", "外資不含外資自營商"}
        IT_NAMES = {"Investment_Trust", "投信"}

        def bucket(name: str) -> str:
            if name in FI_NAMES or "外資" in name or "Foreign" in name:
                return "fi"
            if name in IT_NAMES:
                return "it"
            return "dealer"

        df["bucket"] = df["name"].apply(bucket)
        daily = df.groupby(["date", "bucket"])["net"].sum().unstack(fill_value=0)
        daily = daily.sort_index().tail(days)

        out = {
            "dates":  [d.strftime("%m/%d") for d in daily.index],
            "fi":     [int(round(v)) for v in daily.get("fi",     pd.Series([0] * len(daily))).tolist()],
            "it":     [int(round(v)) for v in daily.get("it",     pd.Series([0] * len(daily))).tolist()],
            "dealer": [int(round(v)) for v in daily.get("dealer", pd.Series([0] * len(daily))).tolist()],
        }
        cache_set(f"inst:{stock_id}", out)
        return out
    except Exception as e:
        print(f"[inst] {stock_id} 失敗: {e}")
        return None


# ----------------------------------------------------------------------------
# 多週期 K 線資料抓取
# ----------------------------------------------------------------------------
PERIOD_CFG = {
    "D": {"period": "1y",  "interval": "1d",  "n": 80,  "label": "日"},
    "W": {"period": "3y",  "interval": "1wk", "n": 80,  "label": "週"},
    "M": {"period": "10y", "interval": "1mo", "n": 80,  "label": "月"},
}


def fetch_stock(code: str, period: str = "D", force: bool = False) -> dict:
    period = period.upper() if period.upper() in PERIOD_CFG else "D"
    cache_key = f"stock:{code}:{period}"
    if not force:
        cached = cache_get(cache_key)
        if cached:
            return cached

    wl = load_watchlist()
    info = wl.get(code)
    if not info:
        raise HTTPException(404, f"未追蹤股票 {code}")

    cfg = PERIOD_CFG[period]
    ticker = yf.Ticker(info["yf"])
    hist = ticker.history(period=cfg["period"], interval=cfg["interval"], auto_adjust=False)
    if hist.empty:
        raise HTTPException(503, f"yfinance 取不到 {code} ({info['yf']}) 資料")
    hist = hist.dropna(subset=["Close"])

    closes = hist["Close"]
    ma5_s, ma20_s, ma60_s = sma(closes, 5), sma(closes, 20), sma(closes, 60)
    rsi_s = rsi_indicator(closes, 14)
    dif_s, dea_s, macd_s = macd_indicator(closes)
    k_s, d_s = kd_indicator(hist, 9)

    last = hist.iloc[-1]
    prev_close = float(hist.iloc[-2]["Close"]) if len(hist) > 1 else float(last["Close"])
    N = cfg["n"]
    recent = hist.iloc[-N:]

    klines = [
        {
            "date": idx.strftime("%m/%d") if period == "D" else idx.strftime("%Y/%m" if period == "M" else "%m/%d"),
            "ohlc": [
                round(float(r["Open"]),  2),
                round(float(r["Close"]), 2),
                round(float(r["Low"]),   2),
                round(float(r["High"]),  2),
            ],
            "volume": int(round(r["Volume"] / 1000)) if r["Volume"] else 0,
        }
        for idx, r in recent.iterrows()
    ]

    def slice_series(s: pd.Series) -> list:
        return [safe(v) for v in s.iloc[-N:].tolist()]

    price = float(last["Close"])
    ma5_v  = safe(ma5_s.iloc[-1])  or price
    ma20_v = safe(ma20_s.iloc[-1]) or price
    ma60_v = safe(ma60_s.iloc[-1]) or price

    if ma5_v > ma20_v > ma60_v:
        ma_status = "多頭排列"
    elif ma5_v < ma20_v < ma60_v:
        ma_status = "空頭排列"
    else:
        ma_status = "盤整"

    if price > ma20_v and price > ma60_v:
        trend = "多頭趨勢"
    elif price < ma20_v and price < ma60_v:
        trend = "空頭趨勢"
    else:
        trend = "盤整"

    avg_vol_5 = float(hist["Volume"].iloc[-5:].mean()) / 1000
    today_vol = float(last["Volume"]) / 1000
    vol_change = (today_vol - avg_vol_5) / avg_vol_5 * 100 if avg_vol_5 > 0 else 0.0

    rsi_v = safe(rsi_s.iloc[-1]) or 50.0
    bias = (price - ma20_v) / ma20_v * 100 if ma20_v else 0

    if rsi_v > 75 and bias > 10:
        risk = "high"
        risk_note = f"RSI {rsi_v:.1f} 超買 + 乖離 {bias:.1f}%，主力疑似出貨。"
        risk_banner = f"高檔過熱警示：RSI {rsi_v:.1f} + 乖離 {bias:.1f}%"
    elif rsi_v > 70 or bias > 8:
        risk = "mid"
        risk_note = f"短線過熱（RSI {rsi_v:.1f}），等待回測。"
        risk_banner = "中度觀察：技術指標過熱"
    elif rsi_v < 30:
        risk = "mid"
        risk_note = f"RSI {rsi_v:.1f} 超賣，可分批承接。"
        risk_banner = "短線超賣，留意止穩"
    else:
        risk = "low"
        risk_note = "技術面健康，籌碼穩定。"
        risk_banner = "趨勢明確 · 操作偏中性"

    span = float(last["High"]) - float(last["Low"])
    if span <= 0:
        outer_pct = 0.5
    else:
        outer_pct = max(0.3, min(0.7, (float(last["Close"]) - float(last["Low"])) / span))
    outer_v = int(round(today_vol * outer_pct))
    inner_v = max(0, int(round(today_vol)) - outer_v)

    high60 = float(recent["High"].max())
    low60  = float(recent["Low"].min())
    if price > ma20_v:
        supports = sorted({int(round(ma20_v)), int(round(ma60_v))})
        resists = sorted({int(round(min(high60, price * 1.05))), int(round(high60))})
    else:
        supports = sorted({int(round(low60)), int(round(ma60_v))})
        resists = sorted({int(round(ma20_v)), int(round(ma5_v))})

    atr = float((hist["High"] - hist["Low"]).iloc[-14:].mean())
    swing = atr if atr > 0 else price * 0.015
    scenario = {
        "up":   {"entry": f"{price:.0f} ~ {price + swing*0.5:.0f}",
                 "sl":    int(round(price - swing * 1.0)),
                 "tp":    f"{int(round(price + swing*2))} / {int(round(price + swing*4))}"},
        "flat": {"entry": f"{price - swing*0.7:.0f} ~ {price - swing*0.2:.0f}",
                 "sl":    int(round(price - swing * 1.5)),
                 "tp":    f"{int(round(price + swing*1))} / {int(round(price + swing*2.5))}"},
        "down": {"entry": f"{price - swing*2:.0f} ~ {price - swing*1.2:.0f}",
                 "sl":    int(round(price - swing * 3)),
                 "tp":    f"{int(round(price - swing*0.3))} / {int(round(price + swing*1))}"},
    }

    inst = fetch_institutional(code) or {"dates": [], "fi": [], "it": [], "dealer": []}
    fi_today = inst["fi"][-1] if inst["fi"] else 0
    it_today = inst["it"][-1] if inst["it"] else 0
    dealer_today = inst["dealer"][-1] if inst["dealer"] else 0
    fi_5  = sum(inst["fi"][-5:])  if inst["fi"] else 0
    fi_10 = sum(inst["fi"])       if inst["fi"] else 0
    it_10 = sum(inst["it"])       if inst["it"] else 0

    bull_count = sum([fi_10 > 0, it_10 > 0])
    if bull_count == 2:
        chip_note = f"外資 {'+' if fi_10>=0 else ''}{fi_10}、投信 {'+' if it_10>=0 else ''}{it_10}（10日）同步買超。"
    elif bull_count == 1:
        chip_note = "三大法人意見分歧，籌碼中性。"
    elif fi_10 < 0 and it_10 < 0:
        chip_note = f"外資 {fi_10}、投信 {it_10}（10日）同步賣超。"
    else:
        chip_note = "尚未取得法人資料。"

    # 訊號（日/週/月皆計算，自動換 lookback 視窗）
    signals = detect_signals(closes, ma5_s, ma20_s, k_s, d_s, rsi_s, hist["High"], hist["Low"], period=period)

    out = {
        "code":   code,
        "name":   info["name"],
        "tag":    info["tag"],
        "group":  info.get("group", "自選"),
        "period": period,
        "price":   round(price, 2),
        "prev":    round(prev_close, 2),
        "open":    round(float(last["Open"]),  2),
        "high":    round(float(last["High"]),  2),
        "low":     round(float(last["Low"]),   2),
        "volume":  int(round(today_vol)),
        "avgVol":  int(round(avg_vol_5)),
        "inner":   inner_v,
        "outer":   outer_v,
        "ma5":  round(ma5_v, 2),
        "ma20": round(ma20_v, 2),
        "ma60": round(ma60_v, 2),
        "rsi":   round(rsi_v, 2),
        "kd_k":  round(safe(k_s.iloc[-1])  or 0, 2),
        "kd_d":  round(safe(d_s.iloc[-1])  or 0, 2),
        "dif":   round(safe(dif_s.iloc[-1]) or 0, 2),
        "dea":   round(safe(dea_s.iloc[-1]) or 0, 2),
        "macd":  round(safe(macd_s.iloc[-1]) or 0, 2),
        "volChange": round(vol_change, 1),
        "trend":     trend,
        "maStatus":  ma_status,
        "risk":       risk,
        "riskNote":   risk_note,
        "riskBanner": risk_banner,
        "resist":  resists,
        "support": supports,
        "klines":      klines,
        "ma5_series":  slice_series(ma5_s),
        "ma20_series": slice_series(ma20_s),
        "ma60_series": slice_series(ma60_s),
        "scenario": scenario,
        "institutional": inst,
        "chip": {
            "fi_today": fi_today, "it_today": it_today, "dealer_today": dealer_today,
            "fi_5": fi_5, "fi_10": fi_10, "it_10": it_10,
            "conclusion": chip_note,
        },
        "signals": signals,
        "asOf":    str(hist.index[-1].date()),
        "source":  "live",
    }
    cache_set(cache_key, out)

    # 順便檢查警示
    if period == "D":
        try:
            check_alert(code, price, prev_close, info["name"], rsi=rsi_v, signals=signals)
        except Exception as e:
            print(f"[alert check] {code}: {e}")

    return out


# ----------------------------------------------------------------------------
# 新聞 (Google News 中文 RSS 為主，yfinance 為備援)
# ----------------------------------------------------------------------------
import xml.etree.ElementTree as _ET
from urllib.parse import quote_plus


def _fetch_google_news_zh(name: str, code: str) -> list:
    """Google News 台灣中文 RSS。免 API key、無速率限制問題。"""
    query = f"{name} {code} 股價 股票"
    url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(query)}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    )
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        root = _ET.fromstring(r.content)
        channel = root.find("channel")
        if channel is None:
            return []
        out = []
        for item in channel.findall("item")[:12]:
            title = item.findtext("title", "") or ""
            link = item.findtext("link", "") or ""
            pub = item.findtext("pubDate", "") or ""
            src_el = item.find("source")
            publisher = (src_el.text if src_el is not None else "") or ""
            # title 通常是「標題 - 來源」格式，分離出純標題
            if " - " in title and not publisher:
                title, publisher = title.rsplit(" - ", 1)
            ts = 0
            if pub:
                try:
                    ts = int(pd.Timestamp(pub).timestamp())
                except Exception:
                    ts = 0
            out.append({
                "title": title.strip(),
                "publisher": publisher.strip() or "Google News",
                "link": link,
                "time": ts,
            })
        return out
    except Exception as e:
        print(f"[google news] {code}: {e}")
        return []


def _fetch_yfinance_news(yf_code: str) -> list:
    try:
        raw = yf.Ticker(yf_code).news or []
        out = []
        for n in raw[:10]:
            content = n.get("content") or n
            title = content.get("title") or n.get("title") or ""
            publisher = (
                (content.get("provider") or {}).get("displayName")
                or n.get("publisher") or ""
            )
            link = (
                (content.get("clickThroughUrl") or {}).get("url")
                or (content.get("canonicalUrl") or {}).get("url")
                or n.get("link") or ""
            )
            ts = n.get("providerPublishTime") or 0
            pub_date = content.get("pubDate", "")
            if not ts and pub_date:
                try:
                    ts = int(pd.Timestamp(pub_date).timestamp())
                except Exception:
                    ts = 0
            out.append({"title": title, "publisher": publisher, "link": link, "time": ts})
        return out
    except Exception as e:
        print(f"[yf news] {yf_code}: {e}")
        return []


def _gemini_sentiment_batch(titles: list[str], key: str) -> list[str]:
    """批次情感分析。回 list of 'positive'/'negative'/'neutral'。"""
    if not titles:
        return []
    try:
        prompt = (
            "判斷以下新聞標題對該股票的情感影響，每行回 positive / negative / neutral 之一，"
            "不要加編號或解釋，順序對齊：\n\n" + "\n".join(titles)
        )
        text = _gemini_call(key, prompt)
        lines = [l.strip().lower() for l in text.split("\n") if l.strip()]
        out = []
        for i in range(len(titles)):
            l = lines[i] if i < len(lines) else ""
            if "pos" in l: out.append("positive")
            elif "neg" in l: out.append("negative")
            else: out.append("neutral")
        return out
    except Exception as e:
        print(f"[sentiment] {e}")
        return ["neutral"] * len(titles)


def fetch_news(code: str) -> list:
    cached = cache_get(f"news:{code}")
    if cached:
        return cached
    wl = load_watchlist()
    info = wl.get(code)
    if not info:
        return []

    # 1) Google News 中文優先
    out = _fetch_google_news_zh(info["name"], code)
    # 2) 中文沒結果才退回 yfinance
    if not out:
        out = _fetch_yfinance_news(info["yf"])

    # 3) 情感分析 (若有 Gemini key)
    key = _get_gemini_key()
    titles = [n.get("title", "") for n in out]
    sentiments = _gemini_sentiment_batch(titles, key) if (key and titles) else ["neutral"] * len(out)
    for i, n in enumerate(out):
        n["sentiment"] = sentiments[i] if i < len(sentiments) else "neutral"

    cache_set_ttl(f"news:{code}", out, 3600)  # 1 小時 — 情感分析不需要 5 分鐘更新
    return out


# ----------------------------------------------------------------------------
# 探測股票代號 (.TW 或 .TWO)
# ----------------------------------------------------------------------------
def probe_yfinance(code: str) -> dict | None:
    code = code.strip()
    for sym in [f"{code}.TW", f"{code}.TWO"]:
        try:
            t = yf.Ticker(sym)
            h = t.history(period="5d", interval="1d", auto_adjust=False)
            if h.empty:
                continue
            try:
                inf = t.info or {}
            except Exception:
                inf = {}
            name = inf.get("longName") or inf.get("shortName") or code
            sector = inf.get("sector") or inf.get("industry") or "—"
            return {
                "code":   code,
                "yf":     sym,
                "name":   name,
                "sector": sector,
                "price":  float(h.iloc[-1]["Close"]),
            }
        except Exception:
            continue
    return None


# ----------------------------------------------------------------------------
# Telegram 警示
# ----------------------------------------------------------------------------
def send_telegram(text: str) -> bool:
    cfg = load_telegram()
    token = cfg.get("bot_token", "")
    chat = cfg.get("chat_id", "")
    if not token or not chat:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, json={
            "chat_id":    chat,
            "text":       text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"[telegram] {e}")
        return False


def check_alert(code: str, price: float, prev: float, name: str = "",
                rsi: float = None, signals: list = None) -> None:
    """檢查警示規則：price / RSI / 訊號事件 / 訊號爆發"""
    alerts = load_alerts()
    rule = alerts.get(code)
    if not rule:
        return
    last_price    = rule.get("last_price", prev)
    last_rsi      = rule.get("last_rsi", rsi if rsi is not None else 50)
    last_sig_keys = set(rule.get("last_sig_keys", []))
    sig_keys = set(s.get("key", "") for s in (signals or []))
    triggered = []

    above = rule.get("above"); below = rule.get("below")
    if above is not None and float(last_price) < float(above) <= price:
        triggered.append(f"🚀 *{name} ({code})* 突破上方警示\n價位: *{above}* → 現價 *{price:.2f}*")
    if below is not None and float(last_price) > float(below) >= price:
        triggered.append(f"⚠️ *{name} ({code})* 跌破下方警示\n價位: *{below}* → 現價 *{price:.2f}*")

    if rsi is not None:
        r_above = rule.get("rsi_above"); r_below = rule.get("rsi_below")
        if r_above is not None and float(last_rsi) < float(r_above) <= rsi:
            triggered.append(f"🔥 *{name} ({code})* RSI 突破 *{r_above}* → 目前 *{rsi:.1f}* (過熱警示)")
        if r_below is not None and float(last_rsi) > float(r_below) >= rsi:
            triggered.append(f"❄️ *{name} ({code})* RSI 跌破 *{r_below}* → 目前 *{rsi:.1f}* (超賣反彈機會)")

    new_signals = sig_keys - last_sig_keys
    if rule.get("on_golden_cross") and "golden_cross" in new_signals:
        triggered.append(f"🌟 *{name} ({code})* 出現 *黃金交叉*")
    if rule.get("on_death_cross") and "death_cross" in new_signals:
        triggered.append(f"💀 *{name} ({code})* 出現 *死亡交叉*")
    if rule.get("on_kd_cross_up") and "kd_cross_up" in new_signals:
        triggered.append(f"🔼 *{name} ({code})* 出現 *KD 黃金交叉*")
    if rule.get("on_breakout") and "breakout_high" in new_signals:
        triggered.append(f"🚀 *{name} ({code})* 突破 *20 日新高*")
    if rule.get("on_breakdown") and "breakdown_low" in new_signals:
        triggered.append(f"📉 *{name} ({code})* 跌破 *20 日新低*")

    burst_n = rule.get("on_signal_burst")
    if burst_n is not None and len(sig_keys) >= int(burst_n) and len(last_sig_keys) < int(burst_n):
        labels = "、".join(s.get("label", "") for s in (signals or [])[:5])
        triggered.append(f"🎯 *{name} ({code})* 訊號爆發！{len(sig_keys)} 個訊號:\n{labels}")

    # Drawdown 警示 (從 60d 高點回檔 X%)
    dd_pct = rule.get("drawdown_pct")
    if dd_pct is not None:
        peak = rule.get("peak_price")
        if peak is None or price > float(peak):
            peak = price
        rule["peak_price"] = peak
        last_dd = rule.get("last_drawdown", 0) or 0
        cur_dd  = (peak - price) / peak * 100 if peak else 0
        if cur_dd >= float(dd_pct) and last_dd < float(dd_pct):
            triggered.append(
                f"🩸 *{name} ({code})* 從 60d 高點回檔 *{cur_dd:.1f}%*\n"
                f"高點 *{peak:.2f}* → 現價 *{price:.2f}* (閾值 {dd_pct}%)"
            )
        rule["last_drawdown"] = cur_dd

    rule["last_price"]    = price
    rule["last_rsi"]      = rsi if rsi is not None else last_rsi
    rule["last_sig_keys"] = list(sig_keys)
    alerts[code] = rule
    save_json(ALERTS_FILE, alerts)
    for msg in triggered:
        send_telegram(msg)
        _append_alert_log(code, name, price, msg)


def _append_alert_log(code: str, name: str, price: float, msg: str) -> None:
    """把觸發的警示寫到 alerts_log.json,給「警示回顧」用。"""
    try:
        log = load_json(ALERTS_LOG_FILE, [])
        kind = "price"
        for prefix, k in [("🚀", "breakout_up"), ("⚠️ ", "breakdown_below"),
                          ("🔥", "rsi_overbought"), ("❄️", "rsi_oversold"),
                          ("🌟", "golden_cross"), ("💀", "death_cross"),
                          ("🔼", "kd_up"), ("📉", "breakdown_low"),
                          ("🎯", "signal_burst"), ("🩸", "drawdown")]:
            if msg.startswith(prefix):
                kind = k; break
        log.append({
            "ts":    time.time(),
            "code":  code,
            "name":  name,
            "price": float(price),
            "kind":  kind,
            "msg":   msg[:200],
        })
        if len(log) > 500:
            log = log[-500:]
        save_json(ALERTS_LOG_FILE, log)
    except Exception as e:
        print(f"[alert_log] {e}")


def alert_worker():
    """背景每 5 分鐘掃描有設警示的股票 + 每天掃族群輪動。"""
    print("[alert_worker] 啟動，每 5 分鐘檢查一次")
    last_group_scan = 0
    while True:
        time.sleep(300)
        try:
            wl = load_watchlist()
            alerts = load_alerts()
            for code in alerts:
                if code not in wl:
                    continue
                try:
                    fetch_stock(code, force=True)  # 內部會 check_alert
                except Exception as e:
                    print(f"[alert_worker] {code}: {e}")
            # 每 24h 掃一次族群輪動 alert
            if time.time() - last_group_scan > 86400:
                try:
                    _scan_group_alerts()
                except Exception as e:
                    print(f"[alert_worker group] {e}")
                last_group_scan = time.time()
        except Exception as e:
            print(f"[alert_worker] {e}")


def _scan_group_alerts():
    """每日掃描族群輪動,達閾值就推 Telegram。
    group_alerts.json:
    {"groups": {"半導體設備": {"ret_1w_above": 5}, ...},
     "_last_pushed": {key: ts}}
    """
    cfg = load_json(GROUP_ALERTS_FILE, {})
    rules = cfg.get("groups", {})
    if not rules:
        return
    last_pushed = cfg.get("_last_pushed", {})
    now_ts = time.time()
    msgs = []
    try:
        data = api_group_rotation()
    except Exception as e:
        print(f"[group_alert] rotation fail: {e}")
        return
    for grp, thresholds in rules.items():
        cur = next((x for x in data if x.get("group") == grp), None)
        if not cur: continue
        for rule, threshold in thresholds.items():
            parts = rule.rsplit("_", 1)
            if len(parts) != 2: continue
            metric, direction = parts
            v = cur.get(metric)
            if v is None: continue
            trig = (direction == "above" and v >= threshold) or \
                   (direction == "below" and v <= threshold)
            if not trig: continue
            dedup_key = f"{grp}:{rule}:{threshold}"
            if now_ts - last_pushed.get(dedup_key, 0) < 86400:
                continue
            arrow = "🚀" if direction == "above" else "⚠️"
            msgs.append(f"{arrow} *族群 {grp}* — {metric} = *{v:+.2f}%* "
                       f"(觸發: {'≥' if direction == 'above' else '≤'} {threshold}%)")
            last_pushed[dedup_key] = now_ts
    if msgs:
        for m in msgs:
            send_telegram(m)
        cfg["_last_pushed"] = last_pushed
        save_json(GROUP_ALERTS_FILE, cfg)
        print(f"[group alert] 推送 {len(msgs)} 則")


# ----------------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------------
app = FastAPI(title="台股情報站 API", version="3.1")

# CORS：允許從 GitHub Pages、file://、其他主機載入的前端訪問本機 server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def fetch_summary(code: str) -> dict:
    """輕量版：只抓最近 30 個交易日，不算複雜指標、不打 FinMind。

    主要給清單頁用，冷快取下也能在 < 5 秒內回應 16 檔。
    若 fetch_stock 已有完整快取則直接重用。
    """
    full = cache_get(f"stock:{code}:D")
    if full:
        return {
            "code":   code,
            "name":   full["name"],
            "tag":    full["tag"],
            "group":  full.get("group", "自選"),
            "price":  full["price"],
            "prev":   full["prev"],
            "asOf":   full["asOf"],
            "signals": full.get("signals", []),
        }

    cached = cache_get(f"summary:{code}")
    if cached:
        return cached

    wl = load_watchlist()
    info = wl.get(code)
    if not info:
        raise HTTPException(404, f"未追蹤股票 {code}")

    try:
        hist = yf.Ticker(info["yf"]).history(period="2mo", interval="1d", auto_adjust=False)
    except Exception as e:
        raise HTTPException(503, f"yfinance 失敗: {e}")
    if hist.empty:
        raise HTTPException(503, f"yfinance 取不到 {code}")

    closes = hist["Close"].dropna()
    last = float(closes.iloc[-1])
    prev = float(closes.iloc[-2]) if len(closes) > 1 else last

    # 訊號：用本地的 30 根日 K 簡算
    ma5_s  = sma(closes, 5)
    ma20_s = sma(closes, 20)
    rsi_s  = rsi_indicator(closes, 14)
    k_s, d_s = kd_indicator(hist, 9)
    sigs = detect_signals(closes, ma5_s, ma20_s, k_s, d_s, rsi_s, hist["High"], hist["Low"], period="D")

    # 短線勝率啟發式
    rsi_v = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else 50.0
    ma5v  = float(ma5_s.iloc[-1])  if not pd.isna(ma5_s.iloc[-1])  else last
    ma20v = float(ma20_s.iloc[-1]) if not pd.isna(ma20_s.iloc[-1]) else last
    base = 50
    if ma5v > ma20v: base += 15
    elif ma5v < ma20v: base -= 15
    if rsi_v > 70: base -= 10
    elif rsi_v < 30: base += 10
    win_rate = max(20, min(85, base))

    out = {
        "code":   code,
        "name":   info["name"],
        "tag":    info["tag"],
        "group":  info.get("group", "自選"),
        "price":  round(last, 2),
        "prev":   round(prev, 2),
        "asOf":   str(hist.index[-1].date()),
        "signals": sigs,
        "rsi":          round(rsi_v, 2),
        "win_rate":     win_rate,
        "signal_count": len(sigs),
    }
    cache_set(f"summary:{code}", out)
    return out


@app.get("/api/stocks")
def api_list():
    """觀察清單摘要 (輕量版，並行抓取)。"""
    from concurrent.futures import ThreadPoolExecutor

    wl = load_watchlist()
    codes = list(wl.keys())

    def safe_summary(code):
        try:
            return fetch_summary(code)
        except Exception as e:
            meta = wl.get(code, {})
            return {
                "code": code, "name": meta.get("name", code), "tag": meta.get("tag", ""),
                "group": meta.get("group", "自選"), "error": str(e),
            }

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(safe_summary, codes))
    return results


@app.get("/api/stock/{code}")
def api_stock(code: str, period: str = "D"):
    try:
        return fetch_stock(code, period=period)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/news/{code}")
def api_news(code: str):
    return fetch_news(code)


@app.get("/api/groups")
def api_groups():
    """族群清單 + 每族群成員代號。"""
    wl = load_watchlist()
    groups: dict[str, list[str]] = {}
    for code, s in wl.items():
        g = s.get("group", "自選")
        groups.setdefault(g, []).append(code)
    return groups


# 4 套權重 prefab (台股版 — 用外資/投信替代估值)
SCORE_WEIGHT_PRESETS = {
    "balanced": {
        "fi_strong": 20, "fi_mid": 12, "fi_sell_strong": -15, "fi_sell_mid": -8,
        "it_strong": 12, "it_mid": 6, "it_sell": -10,
        "win_high": 15, "win_mid": 8,
        "bull_multi": 10, "bull_one": 5, "rs5_strong": 8, "rs5_weak": -8,
        "bear_each": -8, "overheat_each": -5,
        "rsi_overbought": -12, "rsi_oversold": 8,
        "trend_bull": 5, "trend_bear": -5,
    },
    "value": {  # 價值派 — 加重外資 / 籌碼長線
        "fi_strong": 25, "fi_mid": 15, "fi_sell_strong": -20, "fi_sell_mid": -10,
        "it_strong": 20, "it_mid": 10, "it_sell": -15,
        "win_high": 8, "win_mid": 4,
        "bull_multi": 5, "bull_one": 2, "rs5_strong": 4, "rs5_weak": -4,
        "bear_each": -5, "overheat_each": -3,
        "rsi_overbought": -8, "rsi_oversold": 15,
        "trend_bull": 3, "trend_bear": -3,
    },
    "momentum": {  # 動能派 — 加重勝率/訊號/RS
        "fi_strong": 12, "fi_mid": 6, "fi_sell_strong": -8, "fi_sell_mid": -4,
        "it_strong": 8, "it_mid": 4, "it_sell": -6,
        "win_high": 30, "win_mid": 15,
        "bull_multi": 25, "bull_one": 12, "rs5_strong": 20, "rs5_weak": -15,
        "bear_each": -12, "overheat_each": -3,
        "rsi_overbought": -5, "rsi_oversold": 5,
        "trend_bull": 12, "trend_bear": -12,
    },
    "chip": {  # 籌碼派 — 加重外資/投信動向
        "fi_strong": 35, "fi_mid": 18, "fi_sell_strong": -30, "fi_sell_mid": -15,
        "it_strong": 25, "it_mid": 12, "it_sell": -20,
        "win_high": 12, "win_mid": 6,
        "bull_multi": 15, "bull_one": 7, "rs5_strong": 10, "rs5_weak": -8,
        "bear_each": -15, "overheat_each": -5,
        "rsi_overbought": -10, "rsi_oversold": 8,
        "trend_bull": 5, "trend_bear": -8,
    },
}
_SCORE_WEIGHTS = SCORE_WEIGHT_PRESETS["balanced"]


@app.get("/api/ranking")
def api_ranking(by: str = "change", weights: str = "balanced"):
    global _SCORE_WEIGHTS
    _SCORE_WEIGHTS = SCORE_WEIGHT_PRESETS.get(weights, SCORE_WEIGHT_PRESETS["balanced"])
    """熱度榜排序鍵：
    - change / down: 漲幅 / 跌幅
    - volume:        成交量
    - fi / fi_sell:  外資買超 / 賣超
    - rsi / rsi_low: RSI 高 / 低
    - win:           短線勝率
    - signals:       訊號數最多
    - bias:          乖離率絕對值最大
    """
    items = []
    for code in load_watchlist():
        try:
            d = fetch_stock(code)
            change_pct = (d["price"] - d["prev"]) / d["prev"] * 100 if d["prev"] else 0
            sigs = d.get("signals", [])
            ma_status = d.get("maStatus", "")
            rsi_v = d["rsi"]
            base = 50
            if "多頭" in ma_status: base += 15
            elif "空頭" in ma_status: base -= 15
            if rsi_v > 70: base -= 10
            elif rsi_v < 30: base += 10
            win_rate = max(20, min(85, base))
            ma20 = d.get("ma20", 0) or 1
            bias = (d["price"] - ma20) / ma20 * 100 if ma20 else 0
            items.append({
                "code":       code,
                "name":       d["name"],
                "tag":        d["tag"],
                "group":      d.get("group", "自選"),
                "price":      d["price"],
                "prev":       d["prev"],
                "change_pct": round(change_pct, 2),
                "volume":     d["volume"],
                "avgVol":     d["avgVol"],
                "vol_change": d["volChange"],
                "fi_today":   d["chip"]["fi_today"],
                "it_today":   d["chip"]["it_today"],
                "rsi":        rsi_v,
                "trend":      d["trend"],
                "signals":    sigs,
                "signal_count": len(sigs),
                "win_rate":   win_rate,
                "bias":       round(bias, 2),
                "risk":       d["risk"],
            })
        except Exception:
            pass

    # 相對族群強度 RS (1d)
    from statistics import median as _median
    g1d = {}
    for it in items:
        g1d.setdefault(it["group"], []).append(it["change_pct"])
    med1d = {g: _median(vs) for g, vs in g1d.items() if vs}
    for it in items:
        it["rs_1d"] = round(it["change_pct"] - med1d.get(it["group"], 0), 2)

    # 5d / 20d RS — 用 batch yfinance 重抓 60 日 close 算
    try:
        rs_cache = cache_get("rs:5_20")
        if rs_cache is None:
            wl = load_watchlist()
            code_list = list(wl.keys())
            yf_codes_all = [wl[c]["yf"] for c in code_list]
            end_d = pd.Timestamp.today()
            start_d = end_d - pd.Timedelta(days=60)
            data = yf.download(yf_codes_all, start=start_d, end=end_d,
                               auto_adjust=False, progress=False, group_by="ticker", threads=True)
            cc = pd.DataFrame()
            for c, yfc in zip(code_list, yf_codes_all):
                try:
                    col = data[yfc]["Close"] if len(yf_codes_all) > 1 else data["Close"]
                    cc[c] = col
                except Exception:
                    continue
            cc = cc.dropna(how="all")
            rs_cache = {}
            if len(cc) >= 21:
                ret_5d  = ((cc.iloc[-1] - cc.iloc[-6])  / cc.iloc[-6]  * 100).to_dict()
                ret_20d = ((cc.iloc[-1] - cc.iloc[-21]) / cc.iloc[-21] * 100).to_dict()
                rs_cache = {"r5": ret_5d, "r20": ret_20d}
            _cache["rs:5_20"] = (time.time() + 1800 - CACHE_TTL, rs_cache)

        r5 = rs_cache.get("r5", {})
        r20 = rs_cache.get("r20", {})
        g5, g20 = {}, {}
        for it in items:
            v5  = r5.get(it["code"])
            v20 = r20.get(it["code"])
            if v5 is not None and not pd.isna(v5):  g5.setdefault(it["group"], []).append(v5)
            if v20 is not None and not pd.isna(v20): g20.setdefault(it["group"], []).append(v20)
        med5  = {g: _median(vs) for g, vs in g5.items()  if vs}
        med20 = {g: _median(vs) for g, vs in g20.items() if vs}
        for it in items:
            v5  = r5.get(it["code"]); v20 = r20.get(it["code"])
            it["ret_5d"]  = round(float(v5), 2)  if v5  is not None and not pd.isna(v5)  else None
            it["ret_20d"] = round(float(v20), 2) if v20 is not None and not pd.isna(v20) else None
            it["rs_5d"]   = round(it["ret_5d"]  - med5.get(it["group"], 0), 2)  if it["ret_5d"]  is not None else None
            it["rs_20d"]  = round(it["ret_20d"] - med20.get(it["group"], 0), 2) if it["ret_20d"] is not None else None
    except Exception as e:
        print(f"[rs] {e}")
        for it in items:
            it.setdefault("rs_5d", None); it.setdefault("rs_20d", None)

    # 多因子綜合評分 (台股版 — 用外資/投信替代估值)
    W = _SCORE_WEIGHTS
    for it in items:
        score = 0
        reasons_plus = []
        reasons_minus = []
        fi = it.get("fi_today", 0) or 0
        it_t = it.get("it_today", 0) or 0
        wr = it.get("win_rate", 0) or 0
        sigs = it.get("signals", []) or []
        bull = sum(1 for s in sigs if s.get("color") == "red")
        bear = sum(1 for s in sigs if s.get("color") == "green")
        overheat = sum(1 for s in sigs if s.get("color") == "orange")
        rsi_v = it.get("rsi", 50) or 50
        trend = it.get("trend", "")
        rs5 = it.get("rs_5d") or 0

        # 1. 外資
        if fi >= 1000:
            score += W["fi_strong"]; reasons_plus.append(f"外資 +{fi} 張")
        elif fi >= 300:
            score += W["fi_mid"]; reasons_plus.append(f"外資 +{fi} 張")
        elif fi <= -1000:
            score += W["fi_sell_strong"]; reasons_minus.append(f"外資 {fi} 張")
        elif fi <= -300:
            score += W["fi_sell_mid"]; reasons_minus.append(f"外資 {fi} 張")
        # 2. 投信
        if it_t >= 500:
            score += W["it_strong"]; reasons_plus.append(f"投信 +{it_t} 張")
        elif it_t >= 100:
            score += W["it_mid"]; reasons_plus.append(f"投信 +{it_t} 張")
        elif it_t <= -500:
            score += W["it_sell"]; reasons_minus.append(f"投信 {it_t} 張")
        # 3. 勝率
        if wr >= 65:
            score += W["win_high"]; reasons_plus.append(f"勝率 {wr}%")
        elif wr >= 55:
            score += W["win_mid"]; reasons_plus.append(f"勝率 {wr}%")
        # 4. 訊號
        if bull >= 2:
            score += W["bull_multi"]; reasons_plus.append(f"{bull} 多頭訊號")
        elif bull == 1:
            score += W["bull_one"]
        score += W["bear_each"] * bear
        score += W["overheat_each"] * overheat
        if bear: reasons_minus.append(f"{bear} 空頭訊號")
        if overheat: reasons_minus.append(f"{overheat} 過熱警示")
        # 5. RSI
        if rsi_v > 75:
            score += W["rsi_overbought"]; reasons_minus.append(f"RSI {rsi_v:.0f} 超買")
        elif rsi_v < 30:
            score += W["rsi_oversold"]; reasons_plus.append(f"RSI {rsi_v:.0f} 超賣")
        # 6. 趨勢
        if "多頭" in trend:
            score += W["trend_bull"]
        elif "空頭" in trend:
            score += W["trend_bear"]; reasons_minus.append("空頭趨勢")
        # 7. RS 5d
        if rs5 > 5:
            score += W["rs5_strong"]; reasons_plus.append(f"5日比族群 +{rs5:.1f}%")
        elif rs5 < -5:
            score += W["rs5_weak"]; reasons_minus.append(f"5日比族群 {rs5:.1f}%")

        it["score"] = score
        it["score_plus"]  = reasons_plus
        it["score_minus"] = reasons_minus

    def _rs1d(x): return -(x.get("rs_1d")  or -999)
    def _rs5d(x): return -(x.get("rs_5d")  or -999)
    def _rs20(x): return -(x.get("rs_20d") or -999)

    keymap = {
        "change":   lambda x: -x["change_pct"],
        "down":     lambda x:  x["change_pct"],
        "volume":   lambda x: -x["volume"],
        "fi":       lambda x: -x["fi_today"],
        "fi_sell":  lambda x:  x["fi_today"],
        "rsi":      lambda x: -x["rsi"],
        "rsi_low":  lambda x:  x["rsi"],
        "win":      lambda x: -x["win_rate"],
        "signals":  lambda x: -x["signal_count"],
        "bias":     lambda x: -abs(x["bias"]),
        "rs1d":     _rs1d,
        "rs5d":     _rs5d,
        "rs20d":    _rs20,
        "score":    lambda x: -x["score"],
    }
    items.sort(key=keymap.get(by, keymap["change"]))
    return items


# ----- 觀察清單管理 -----
class AddStockReq(BaseModel):
    code:  str
    name:  Optional[str] = None
    tag:   Optional[str] = None
    group: Optional[str] = "自選"


@app.get("/api/watchlist")
def api_get_watchlist():
    return load_watchlist()


@app.post("/api/watchlist")
def api_add_watchlist(req: AddStockReq):
    code = req.code.strip()
    if not code.isdigit():
        raise HTTPException(400, "代號需為數字")
    wl = load_watchlist()
    if code in wl:
        return {"ok": True, "msg": "已在觀察清單", "data": wl[code]}
    probe = probe_yfinance(code)
    if not probe:
        raise HTTPException(404, f"yfinance 找不到 {code}（試過 .TW / .TWO）")
    name = req.name or probe["name"]
    if len(name) > 12:
        name = name[:12]
    wl[code] = {
        "name":  name,
        "tag":   req.tag or probe.get("sector", "—"),
        "yf":    probe["yf"],
        "group": req.group or "自選",
    }
    save_json(WATCHLIST_FILE, wl)
    # 清掉清單快取，下次列表會重撈
    _cache.pop(f"stock:{code}:D", None)
    return {"ok": True, "data": wl[code], "probe": probe}


@app.delete("/api/watchlist/{code}")
def api_del_watchlist(code: str):
    wl = load_watchlist()
    if code not in wl:
        raise HTTPException(404)
    del wl[code]
    save_json(WATCHLIST_FILE, wl)
    return {"ok": True}


@app.get("/api/probe/{code}")
def api_probe(code: str):
    p = probe_yfinance(code)
    if not p:
        raise HTTPException(404, "yfinance 找不到")
    return p


# ----- Telegram -----
class TelegramReq(BaseModel):
    bot_token: str
    chat_id:   str


@app.post("/api/telegram")
def api_telegram_set(req: TelegramReq):
    save_json(TELEGRAM_FILE, {
        "bot_token": req.bot_token.strip(),
        "chat_id":   req.chat_id.strip(),
    })
    ok = send_telegram(f"✅ *台股情報站* Telegram 連線測試成功\n時間: `{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}`")
    return {"ok": True, "test_sent": ok}


@app.get("/api/telegram")
def api_telegram_get():
    cfg = load_telegram()
    return {
        "configured": bool(cfg.get("bot_token") and cfg.get("chat_id")),
        "chat_id":    cfg.get("chat_id", ""),
    }


# ----- Alerts -----
class AlertReq(BaseModel):
    above: Optional[float] = None
    below: Optional[float] = None
    rsi_above: Optional[float] = None
    rsi_below: Optional[float] = None
    on_golden_cross: Optional[bool] = None
    on_death_cross:  Optional[bool] = None
    on_kd_cross_up:  Optional[bool] = None
    on_breakout:     Optional[bool] = None
    on_breakdown:    Optional[bool] = None
    on_signal_burst: Optional[int]  = None


@app.get("/api/alerts")
def api_get_alerts():
    return load_alerts()


@app.post("/api/alerts/{code}")
def api_set_alert(code: str, req: AlertReq):
    alerts = load_alerts()
    rule = alerts.get(code, {})
    for field in ["above", "below", "rsi_above", "rsi_below",
                  "on_golden_cross", "on_death_cross", "on_kd_cross_up",
                  "on_breakout", "on_breakdown", "on_signal_burst"]:
        val = getattr(req, field, None)
        if val is not None:
            rule[field] = val
        else:
            rule.pop(field, None)
    if "last_price" not in rule:
        try:
            d = fetch_stock(code)
            rule["last_price"] = d["price"]
            rule["last_rsi"]   = d.get("rsi", 50)
            rule["last_sig_keys"] = [s["key"] for s in d.get("signals", [])]
        except Exception:
            rule["last_price"] = 0
    alerts[code] = rule
    save_json(ALERTS_FILE, alerts)
    return {"ok": True, "data": rule}


@app.delete("/api/alerts/{code}")
def api_del_alert(code: str):
    alerts = load_alerts()
    alerts.pop(code, None)
    save_json(ALERTS_FILE, alerts)
    return {"ok": True}


# ----- Misc -----
@app.get("/api/refresh")
def api_refresh():
    _cache.clear()
    return {"ok": True, "msg": "快取已清除"}


@app.post("/api/seed-defaults")
def api_seed_defaults():
    """把 DEFAULT_WATCHLIST 中還沒在 watchlist.json 的股票補進去。"""
    wl = load_watchlist()
    added = []
    for code, meta in DEFAULT_WATCHLIST.items():
        if code not in wl:
            wl[code] = dict(meta)
            added.append(code)
    if added:
        save_json(WATCHLIST_FILE, wl)
        for k in list(_cache.keys()):
            if k.startswith("summary:") or k.startswith("stock:"):
                _cache.pop(k, None)
    return {"ok": True, "added": added, "n_added": len(added),
            "msg": f"已新增 {len(added)} 檔到觀察清單"}


# ============================================================================
# 1) Portfolio (個人持股；同代號可多筆，各自有 id)
# ============================================================================
import uuid as _uuid


def load_portfolio() -> list:
    """永遠回 list。舊版 dict 格式 (code → holding) 會自動遷移成新版 list。"""
    raw = load_json(PORTFOLIO_FILE, [])
    if isinstance(raw, dict):
        migrated = []
        for code, h in raw.items():
            migrated.append({
                "id":         _uuid.uuid4().hex[:12],
                "code":       code,
                "shares":     float(h.get("shares", 0) or 0),
                "cost_price": float(h.get("cost_price", 0) or 0),
                "buy_date":   h.get("buy_date", "") or "",
                "note":       h.get("note", "") or "",
            })
        save_json(PORTFOLIO_FILE, migrated)
        print(f"[portfolio] 舊 dict 格式遷移成 list，{len(migrated)} 筆")
        return migrated
    return raw if isinstance(raw, list) else []


class HoldingReq(BaseModel):
    code:       str
    shares:     float       # 張 (1 張 = 1000 股)
    cost_price: float
    buy_date:   Optional[str] = None
    note:       Optional[str] = ""


def _trailing_stop_advice(yf_code: str, buy_date: str, cost_price: float, current_price: float) -> dict:
    """計算移動停利建議 (台股版)：高點 -10% 移動停利、保本停利、+20% 目標。"""
    try:
        kwargs = {"period": "1y", "interval": "1d", "auto_adjust": False}
        if buy_date:
            kwargs = {"start": buy_date, "interval": "1d", "auto_adjust": False}
        hist = yf.Ticker(yf_code).history(**kwargs)
        if hist.empty:
            return {"trail_stop": None, "post_high": None, "breakeven": None,
                    "target": round(cost_price * 1.20, 2), "rule": "資料不足"}
        post_high = float(hist["High"].max())
    except Exception as e:
        print(f"[trail] {yf_code}: {e}")
        return {"trail_stop": None, "post_high": None, "breakeven": None,
                "target": round(cost_price * 1.20, 2), "rule": "計算失敗"}

    trail_pct  = 0.10
    trail_stop = round(post_high * (1 - trail_pct), 2)
    target     = round(cost_price * 1.20, 2)
    breakeven  = round(cost_price, 2) if current_price >= cost_price * 1.05 else None
    ret_pct    = (current_price - cost_price) / cost_price * 100 if cost_price else 0

    if ret_pct >= 20:
        rule = f"已 +{ret_pct:.0f}% 達目標，建議分批停利"
    elif current_price <= trail_stop:
        rule = f"已觸發 -10% 移動停利 ({trail_stop})，建議出場"
    elif breakeven:
        rule = f"已保本；停利上移至高點 -10% = {trail_stop}"
    else:
        rule = f"持有觀察，停損 {round(cost_price * 0.92, 2)} (-8%)"

    return {
        "trail_stop": trail_stop,
        "post_high":  round(post_high, 2),
        "breakeven":  breakeven,
        "target":     target,
        "rule":       rule,
    }


@app.get("/api/portfolio")
def api_get_portfolio():
    """回傳所有持股 + 市值/損益/移動停利建議。台股 1 張 = 1000 股。"""
    p = load_portfolio()
    if not p:
        return {"holdings": [], "summary": {"total_cost": 0, "total_value": 0,
                                              "total_pnl": 0, "total_pnl_pct": 0, "count": 0}}
    holdings = []
    total_cost = 0.0
    total_value = 0.0
    wl = load_watchlist()
    for h in p:
        code = h["code"]
        try:
            s = fetch_summary(code)
            price = float(s["price"])
            name  = s["name"]
            tag   = s.get("tag", "")
        except Exception:
            price, name, tag = float(h["cost_price"]), code, ""
        yf_code = wl.get(code, {}).get("yf", code + ".TW")
        shares = float(h["shares"])
        cost   = shares * 1000 * float(h["cost_price"])
        value  = shares * 1000 * price
        pnl    = value - cost
        pnl_pct = (price - float(h["cost_price"])) / float(h["cost_price"]) * 100 if h["cost_price"] else 0
        total_cost += cost
        total_value += value

        trail = _trailing_stop_advice(yf_code, h.get("buy_date", ""), float(h["cost_price"]), price)

        holdings.append({
            "id":         h.get("id") or _uuid.uuid4().hex[:12],
            "code":       code,
            "name":       name,
            "tag":        tag,
            "shares":     shares,
            "cost_price": float(h["cost_price"]),
            "buy_date":   h.get("buy_date", ""),
            "note":       h.get("note", ""),
            "current":    round(price, 2),
            "cost":       round(cost, 0),
            "value":      round(value, 0),
            "pnl":        round(pnl, 0),
            "pnl_pct":    round(pnl_pct, 2),
            "trail_stop": trail.get("trail_stop"),
            "post_high":  trail.get("post_high"),
            "breakeven":  trail.get("breakeven"),
            "target":     trail.get("target"),
            "rule":       trail.get("rule"),
        })
    for h in holdings:
        h["weight"] = round(h["value"] / total_value * 100, 1) if total_value > 0 else 0
    # 預設按名稱字母排（前端可再以欄位點擊重新排序）
    holdings.sort(key=lambda x: (x.get("name") or x.get("code") or "").lower())

    summary = {
        "total_cost":    round(total_cost, 0),
        "total_value":   round(total_value, 0),
        "total_pnl":     round(total_value - total_cost, 0),
        "total_pnl_pct": round((total_value - total_cost) / total_cost * 100, 2) if total_cost > 0 else 0,
        "count":         len(holdings),
    }
    return {"holdings": holdings, "summary": summary}


@app.post("/api/portfolio")
def api_add_portfolio(req: HoldingReq):
    code = req.code.strip()
    if not code.isdigit():
        raise HTTPException(400, "代號需為數字")
    if req.shares <= 0 or req.cost_price <= 0:
        raise HTTPException(400, "張數與成本價需 > 0")

    p = load_portfolio()  # list
    wl = load_watchlist()
    added_to_watchlist = False
    if code not in wl:
        probe = probe_yfinance(code)
        if probe:
            wl[code] = {
                "name":  probe["name"][:12],
                "tag":   probe.get("sector", "—"),
                "yf":    probe["yf"],
                "group": "持股",
            }
            save_json(WATCHLIST_FILE, wl)
            added_to_watchlist = True
            for k in list(_cache.keys()):
                if k.startswith("summary:") or k.startswith("stock:"):
                    _cache.pop(k, None)

    new_holding = {
        "id":         _uuid.uuid4().hex[:12],
        "code":       code,
        "shares":     float(req.shares),
        "cost_price": float(req.cost_price),
        "buy_date":   req.buy_date or "",
        "note":       req.note or "",
    }
    p.append(new_holding)
    save_json(PORTFOLIO_FILE, p)
    return {"ok": True, "data": new_holding, "added_to_watchlist": added_to_watchlist}


@app.delete("/api/portfolio/{holding_id}")
def api_del_portfolio(holding_id: str):
    p = load_portfolio()
    new_p = [h for h in p if h.get("id") != holding_id]
    if len(new_p) == len(p):
        new_p = [h for h in p if h.get("code") != holding_id]
        if len(new_p) == len(p):
            raise HTTPException(404)
    save_json(PORTFOLIO_FILE, new_p)
    return {"ok": True, "removed": len(p) - len(new_p)}


# ============================================================================
# 2) Signal backtest (訊號績效回測)
# ============================================================================
@app.get("/api/signal-stats/{code}/{signal_key}")
def api_signal_stats(code: str, signal_key: str, forward: int = 5):
    """掃過去 2 年該訊號出現的歷史，計算後 N 天平均報酬與勝率。"""
    cache_key = f"sigstat:{code}:{signal_key}:{forward}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    wl = load_watchlist()
    info = wl.get(code)
    if not info:
        raise HTTPException(404)

    try:
        hist = yf.Ticker(info["yf"]).history(period="2y", interval="1d", auto_adjust=False)
    except Exception as e:
        raise HTTPException(503, str(e))
    if len(hist) < 60:
        raise HTTPException(503, "歷史資料不足")
    hist = hist.dropna(subset=["Close"])

    closes = hist["Close"]
    ma5_s, ma20_s = sma(closes, 5), sma(closes, 20)
    rsi_s = rsi_indicator(closes, 14)
    k_s, d_s = kd_indicator(hist, 9)

    occurrences = []
    for i in range(60, len(hist) - forward):
        snap_close = closes.iloc[: i + 1]
        snap_ma5   = ma5_s.iloc[: i + 1]
        snap_ma20  = ma20_s.iloc[: i + 1]
        snap_rsi   = rsi_s.iloc[: i + 1]
        snap_k     = k_s.iloc[: i + 1]
        snap_d     = d_s.iloc[: i + 1]
        snap_h     = hist["High"].iloc[: i + 1]
        snap_l     = hist["Low"].iloc[: i + 1]
        sigs = detect_signals(snap_close, snap_ma5, snap_ma20, snap_k, snap_d,
                              snap_rsi, snap_h, snap_l, period="D")
        if any(s["key"] == signal_key for s in sigs):
            entry = float(closes.iloc[i])
            future = float(closes.iloc[i + forward])
            ret = (future - entry) / entry * 100
            occurrences.append({
                "date":   hist.index[i].strftime("%Y-%m-%d"),
                "entry":  round(entry, 2),
                "future": round(future, 2),
                "return": round(ret, 2),
            })

    if not occurrences:
        result = {"signal": signal_key, "count": 0, "msg": "歷史中無此訊號"}
    else:
        rets = [o["return"] for o in occurrences]
        win = sum(1 for r in rets if r > 0)
        result = {
            "signal":      signal_key,
            "code":        code,
            "name":        info["name"],
            "forward_days": forward,
            "count":       len(occurrences),
            "win_rate":    round(win / len(rets) * 100, 1),
            "avg_return":  round(sum(rets) / len(rets), 2),
            "best":        round(max(rets), 2),
            "worst":       round(min(rets), 2),
            "occurrences": occurrences[-15:],
        }
    cache_set(cache_key, result)
    return result


# ============================================================================
# 3) AI commentary (Gemini)
# ============================================================================
GEMINI_MODELS = ("gemini-2.5-flash", "gemini-2.5-flash-lite")


def _gemini_call(key: str, prompt: str) -> str:
    """新版 google-genai SDK；依序嘗試多個模型，回傳首個成功結果文字。"""
    from google import genai
    client = genai.Client(api_key=key)
    last_err = None
    for model_name in GEMINI_MODELS:
        try:
            resp = client.models.generate_content(model=model_name, contents=prompt)
            txt = (resp.text or "").strip()
            if txt:
                return txt
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("Gemini all models failed")


def _get_gemini_key() -> str:
    cfg = load_json(GEMINI_FILE, {})
    return (cfg.get("api_key", "") or os.environ.get("GEMINI_API_KEY", "")).strip()


class GeminiReq(BaseModel):
    api_key: str


@app.post("/api/gemini")
def api_gemini_set(req: GeminiReq):
    save_json(GEMINI_FILE, {"api_key": req.api_key.strip()})
    return {"ok": True}


@app.get("/api/gemini")
def api_gemini_get():
    return {"configured": bool(_get_gemini_key())}


@app.get("/api/ai-comment/{code}")
def api_ai_comment(code: str):
    """用 Gemini 對單檔產生 3-5 句中文評論。"""
    key = _get_gemini_key()
    if not key:
        return {"ok": False, "msg": "尚未設定 GEMINI API key（從工具列「🤖 AI」按鈕設定）"}

    cache_key = f"ai:{code}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    try:
        d = fetch_stock(code)
    except Exception as e:
        raise HTTPException(503, str(e))

    portfolio = load_portfolio()  # list (multi-entry)
    holdings_of_code = [h for h in portfolio if h.get("code") == code]
    wl = load_watchlist()

    sigs = "、".join(s["label"] for s in d.get("signals", [])) or "無強烈訊號"

    # 月營收 YoY (最近 3 個月)
    fund_line = ""
    try:
        from concurrent.futures import ThreadPoolExecutor
        fund = api_fundamentals(code) if False else None  # 避開 endpoint 直接重用 cache 路徑
        # 直接呼 cache 同樣的函式 (api_fundamentals 本身就有 cache)
        # 為避免循環呼叫風險，這裡走 yfinance? 算了，直接從 FinMind 抓即可
        cached_f = cache_get(f"fund:{code}")
        if cached_f and cached_f.get("revenue"):
            recent = cached_f["revenue"][-3:]
            yoy_strs = []
            for r in recent:
                yoy = f"YoY {r['yoy']:+.1f}%" if r.get("yoy") is not None else "—"
                rev_b = r["revenue"] / 1e8
                yoy_strs.append(f"{r['ym']} {rev_b:.0f}億 {yoy}")
            fund_line = f"\n月營收近 3 月：{' / '.join(yoy_strs)}"
    except Exception:
        pass

    # 使用者持股 + 移動停利建議
    pos = ""
    if holdings_of_code:
        total_shares = sum(float(h.get("shares", 0)) for h in holdings_of_code)
        total_cost   = sum(float(h.get("shares", 0)) * float(h.get("cost_price", 0))
                           for h in holdings_of_code)
        avg_cost = total_cost / total_shares if total_shares > 0 else 0
        ret = (d["price"] - avg_cost) / avg_cost * 100 if avg_cost else 0
        n = len(holdings_of_code)
        yf_code = wl.get(code, {}).get("yf", code + ".TW")
        trail = _trailing_stop_advice(yf_code, "", avg_cost, d["price"])
        pos = (f"\n使用者持股：{total_shares} 張 ({n} 筆)，平均成本 {avg_cost:.2f}，"
               f"損益 {ret:+.2f}%，移動停利建議：{trail.get('rule','—')}")

    prompt = f"""你是台股技術分析助理。用 4-6 句繁體中文評論以下個股，最後給「短線操作建議」一句話。
請避免免責聲明、不要列點，直接給結論。評論時請綜合技術面 + 籌碼面 (三大法人) + 基本面 (月營收)。

【{d['name']} ({d['code']}) {d['tag']}】
收盤 {d['price']}（前日 {d['prev']}, {(d['price']-d['prev'])/d['prev']*100:+.2f}%）
趨勢：{d['trend']}, 均線：{d['maStatus']}（5/20/60 = {d['ma5']}/{d['ma20']}/{d['ma60']}）
RSI(14) = {d['rsi']}, KD(9,3) K/D = {d['kd_k']}/{d['kd_d']}, MACD {d['macd']}
量能變化 {d['volChange']:+.1f}%（5日均量 {d['avgVol']} 張）
外資 10 日累計 {d['chip']['fi_10']:+d} 張、投信 {d['chip']['it_10']:+d} 張
近期訊號：{sigs}
壓力 {d['resist']} / 支撐 {d['support']}{fund_line}{pos}
"""
    try:
        text = _gemini_call(key, prompt)
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "spending cap" in msg:
            return {"ok": False,
                    "msg": "Gemini 月度花費上限到了。請去 https://ai.studio/spend 拉高 cap, "
                           "或換另一支 API key (不同 project),或等下個月重置。"
                           "提示:評論已 cache 12 小時、新聞 1 小時,正常使用每月應該很少打中上限。"}
        return {"ok": False, "msg": f"Gemini 失敗: {e}"}

    out = {"ok": True, "code": code, "comment": text, "asOf": d["asOf"]}
    cache_set_ttl(cache_key, out, 43200)  # 12 小時 — 評論不會分鐘級變化
    return out


# ============================================================================
# 4) Group heatmap (族群熱度地圖)
# ============================================================================
@app.get("/api/group-heatmap")
def api_group_heatmap():
    from concurrent.futures import ThreadPoolExecutor
    wl = load_watchlist()

    def safe(code):
        try:
            return fetch_summary(code)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = [r for r in ex.map(safe, wl.keys()) if r and "error" not in r]

    groups: dict[str, list] = {}
    for s in results:
        groups.setdefault(s.get("group", "自選"), []).append(s)

    out = []
    for g, members in groups.items():
        changes = [(m["price"] - m["prev"]) / m["prev"] * 100 for m in members if m.get("prev")]
        avg = sum(changes) / len(changes) if changes else 0
        wins = sum(1 for c in changes if c > 0)
        members_sorted = sorted(members, key=lambda x: -((x["price"] - x["prev"]) / x["prev"] * 100 if x.get("prev") else 0))
        out.append({
            "group":      g,
            "count":      len(members),
            "avg_change": round(avg, 2),
            "wins":       wins,
            "losses":     len(changes) - wins,
            "max_up":     round(max(changes), 2) if changes else 0,
            "max_down":   round(min(changes), 2) if changes else 0,
            "members": [{
                "code":   m["code"],
                "name":   m["name"],
                "price":  m["price"],
                "change": round((m["price"] - m["prev"]) / m["prev"] * 100, 2) if m.get("prev") else 0,
            } for m in members_sorted],
        })
    out.sort(key=lambda x: -x["avg_change"])
    return out


@app.get("/api/group-rotation")
def api_group_rotation():
    """族群輪動偵測 (台股版)：每族群 1W/1M/3M 平均報酬 + 動量加速度。"""
    cache_key = "rotation:90d"
    cached = cache_get(cache_key)
    if cached:
        return cached

    wl = load_watchlist()
    codes = list(wl.keys())
    yf_codes = [wl[c]["yf"] for c in codes]
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=120)
    try:
        data = yf.download(yf_codes, start=start, end=end, auto_adjust=False,
                           progress=False, group_by="ticker", threads=True)
    except Exception as e:
        raise HTTPException(503, f"yfinance batch fail: {e}")

    closes = pd.DataFrame()
    for c, yfc in zip(codes, yf_codes):
        try:
            col = data[yfc]["Close"] if len(yf_codes) > 1 else data["Close"]
            closes[c] = col
        except Exception:
            continue
    closes = closes.dropna(how="all")
    if len(closes) < 30:
        raise HTTPException(503, "資料不足")

    def n_day_ret(n):
        if len(closes) < n + 1:
            return {}
        return ((closes.iloc[-1] - closes.iloc[-n-1]) / closes.iloc[-n-1] * 100).to_dict()

    ret_1w = n_day_ret(5)
    ret_1m = n_day_ret(20)
    ret_3m = n_day_ret(60) if len(closes) >= 61 else {}

    by_group: dict[str, list[str]] = {}
    for code in closes.columns:
        g = wl.get(code, {}).get("group", "其他")
        by_group.setdefault(g, []).append(code)

    from statistics import mean as _mean
    rotation = []
    for g, members in by_group.items():
        def clean(d):
            return [v for v in (d.get(c) for c in members) if v is not None and not pd.isna(v)]
        r1w_v = clean(ret_1w); r1m_v = clean(ret_1m); r3m_v = clean(ret_3m)
        r1w = _mean(r1w_v) if r1w_v else 0
        r1m = _mean(r1m_v) if r1m_v else 0
        r3m = _mean(r3m_v) if r3m_v else 0
        momentum = round(r1w - (r1m / 4), 2) if r1m_v and r1w_v else 0
        rotation.append({
            "group": g, "n": len(members),
            "ret_1w": round(r1w, 2), "ret_1m": round(r1m, 2), "ret_3m": round(r3m, 2),
            "momentum": momentum, "members": members,
        })
    rotation.sort(key=lambda x: -x["momentum"])
    cache_set(cache_key, rotation)
    return rotation


# ============================================================================
# 市場寬度 (Breadth) — 台股版
# ============================================================================
@app.get("/api/breadth")
def api_breadth():
    """watchlist 寬度 + 加權指數 vs 0050/0056 比較。"""
    cache_key = "breadth"
    cached = cache_get(cache_key)
    if cached:
        return cached

    wl = load_watchlist()
    above_20 = above_60 = 0
    advancers = decliners = 0
    total = 0
    bull_trend = bear_trend = 0
    for code in wl:
        try:
            d = fetch_stock(code)
        except Exception:
            continue
        total += 1
        price = d["price"]
        ma20  = d.get("ma20") or 0
        ma60  = d.get("ma60") or 0
        prev  = d.get("prev") or 0
        if ma20 and price > ma20: above_20 += 1
        if ma60 and price > ma60: above_60 += 1
        if prev:
            if price > prev: advancers += 1
            elif price < prev: decliners += 1
        if "多頭" in d.get("trend", ""): bull_trend += 1
        elif "空頭" in d.get("trend", ""): bear_trend += 1

    pct_20 = round(above_20 / total * 100, 1) if total else 0
    pct_60 = round(above_60 / total * 100, 1) if total else 0
    ad_ratio = round(advancers / decliners, 2) if decliners else (advancers if advancers else 0)

    # ^TWII (加權) vs 0050.TW (大盤 ETF) vs 0056.TW (高股息中型股) 對比
    twii_chg = etf50_chg = etf56_chg = None
    twii_vs_56 = None
    try:
        end = pd.Timestamp.today()
        start = end - pd.Timedelta(days=10)
        data = yf.download(["^TWII", "0050.TW", "0056.TW"], start=start, end=end,
                           auto_adjust=False, progress=False, group_by="ticker", threads=True)
        def _last_chg(ticker):
            try:
                c = data[ticker]["Close"]
                if len(c) >= 2:
                    return float((c.iloc[-1] - c.iloc[-2]) / c.iloc[-2] * 100)
            except Exception:
                return None
            return None
        twii_chg  = _last_chg("^TWII")
        etf50_chg = _last_chg("0050.TW")
        etf56_chg = _last_chg("0056.TW")
        if twii_chg is not None and etf56_chg is not None:
            # 加權拉但 0056 沒跟 → 大型權值股拉動,中型股弱
            twii_vs_56 = round(twii_chg - etf56_chg, 2)
    except Exception as e:
        print(f"[breadth twii] {e}")

    status = "neutral"
    note = ""
    if pct_20 >= 70 and pct_60 >= 60:
        status = "strong"; note = "多頭格局明確"
    elif pct_20 >= 50 and pct_60 >= 50:
        status = "healthy"; note = "偏多但留意過熱"
    elif pct_20 < 40 and pct_60 < 40:
        status = "weak"; note = "短中線都失守,警戒"
    elif pct_60 < 50 and pct_20 > 60:
        status = "divergence"; note = "短線拉、中線弱,假反彈警惕"

    out = {
        "total":         total,
        "pct_above_50":  pct_20,   # 鏡像 US 欄位名,以便共用前端代碼
        "pct_above_200": pct_60,
        "advancers":     advancers,
        "decliners":     decliners,
        "unchanged":     total - advancers - decliners,
        "ad_ratio":      ad_ratio,
        "bull_trend":    bull_trend,
        "bear_trend":    bear_trend,
        "spy_change":    round(twii_chg,  2) if twii_chg  is not None else None,
        "rsp_change":    round(etf56_chg, 2) if etf56_chg is not None else None,
        "spy_vs_rsp":    twii_vs_56,
        "status":        status,
        "note":          note,
    }
    cache_set(cache_key, out)
    return out


# ============================================================================
# 52 週新高 / 新低掃描 — 台股版
# ============================================================================
@app.get("/api/52w-scan")
def api_52w_scan():
    """掃描 watchlist 創 52 週新高/新低的個股。"""
    cache_key = "52w_scan"
    cached = cache_get(cache_key)
    if cached:
        return cached

    wl = load_watchlist()
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=400)
    yf_codes = [wl[c]["yf"] for c in wl]
    codes = list(wl.keys())
    new_highs = []
    new_lows = []
    near_highs = []
    try:
        data = yf.download(yf_codes, start=start, end=end, auto_adjust=False,
                           progress=False, group_by="ticker", threads=True)
    except Exception as e:
        raise HTTPException(503, f"yfinance batch fail: {e}")

    for code, yfc in zip(codes, yf_codes):
        try:
            df = data[yfc] if len(yf_codes) > 1 else data
            close = df["Close"].dropna()
            vol   = df["Volume"].dropna()
            if len(close) < 252:
                continue
            year = close.iloc[-252:]
            cur = float(close.iloc[-1])
            prev = float(close.iloc[-2]) if len(close) >= 2 else cur
            year_high = float(year.max())
            year_low  = float(year.min())
            cur_vol = float(vol.iloc[-1]) if len(vol) else 0
            avg_vol_20 = float(vol.iloc[-20:].mean()) if len(vol) >= 20 else 0
            vol_ratio = round(cur_vol / avg_vol_20, 2) if avg_vol_20 else 0
            chg_pct = round((cur - prev) / prev * 100, 2) if prev else 0
            info = wl.get(code, {})
            base = {
                "code":       code,
                "name":       info.get("name", code),
                "group":      info.get("group", "—"),
                "price":      round(cur, 2),
                "year_high":  round(year_high, 2),
                "year_low":   round(year_low, 2),
                "vol_ratio":  vol_ratio,
                "change_pct": chg_pct,
            }
            if cur >= year_high * 0.999:
                new_highs.append({**base, "type": "new_high"})
            elif cur <= year_low * 1.001:
                new_lows.append({**base, "type": "new_low"})
            elif cur >= year_high * 0.97:
                base["pct_from_high"] = round((cur - year_high) / year_high * 100, 2)
                near_highs.append({**base, "type": "near_high"})
        except Exception:
            continue

    new_highs.sort(key=lambda x: -x["vol_ratio"])
    near_highs.sort(key=lambda x: -(x.get("pct_from_high") or -999))
    out = {
        "new_highs":  new_highs,
        "new_lows":   new_lows,
        "near_highs": near_highs[:10],
        "as_of":      str(end.date()),
    }
    cache_set(cache_key, out)
    return out


# ============================================================================
# 警示觸發紀錄 — 過去 N 天 + 每筆後續走勢
# ============================================================================
@app.get("/api/alerts-log")
def api_alerts_log(days: int = 30):
    """過去 N 天觸發的警示 + 每筆 1d/5d/20d 後續走勢回顧。"""
    cache_key = f"alerts_log:{days}"
    cached = cache_get(cache_key)
    if cached:
        return cached
    log = load_json(ALERTS_LOG_FILE, [])
    if not log:
        return {"entries": [], "stats": {}, "asOf": str(pd.Timestamp.today().date())}
    cutoff_ts = time.time() - days * 86400
    recent = [e for e in log if e.get("ts", 0) >= cutoff_ts]
    if not recent:
        return {"entries": [], "stats": {}, "asOf": str(pd.Timestamp.today().date())}

    wl = load_watchlist()
    codes = sorted({e["code"] for e in recent if e["code"] in wl})
    yf_codes = [wl[c]["yf"] for c in codes]
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=days + 60)
    closes = {}
    if yf_codes:
        try:
            data = yf.download(yf_codes, start=start, end=end, auto_adjust=False,
                               progress=False, group_by="ticker", threads=True)
            for c, yfc in zip(codes, yf_codes):
                try:
                    closes[c] = data[yfc]["Close"].dropna() if len(yf_codes) > 1 else data["Close"].dropna()
                except Exception:
                    continue
        except Exception as e:
            print(f"[alerts_log yf] {e}")

    out_entries = []
    for e in reversed(recent):
        c = e["code"]
        trigger_price = float(e.get("price", 0))
        ts = e.get("ts", 0)
        dt = pd.Timestamp(ts, unit="s")
        ret_1d = ret_5d = ret_20d = None
        c_series = closes.get(c)
        if c_series is not None and len(c_series) > 0:
            try:
                after = c_series[c_series.index >= dt.tz_localize(None) if c_series.index.tz else c_series.index >= dt]
                if len(after) >= 2:
                    ret_1d = round((float(after.iloc[1]) - trigger_price) / trigger_price * 100, 2) if trigger_price else None
                if len(after) >= 6:
                    ret_5d = round((float(after.iloc[5]) - trigger_price) / trigger_price * 100, 2) if trigger_price else None
                if len(after) >= 21:
                    ret_20d = round((float(after.iloc[20]) - trigger_price) / trigger_price * 100, 2) if trigger_price else None
            except Exception:
                pass
        out_entries.append({
            "ts": int(ts), "date": dt.strftime("%Y-%m-%d %H:%M"),
            "code": c, "name": e.get("name", c), "price": trigger_price,
            "kind": e.get("kind", "—"), "msg": e.get("msg", ""),
            "ret_1d": ret_1d, "ret_5d": ret_5d, "ret_20d": ret_20d,
        })

    stats = {}
    for k in {x["kind"] for x in out_entries}:
        same = [x for x in out_entries if x["kind"] == k]
        wins_5d = sum(1 for x in same if x["ret_5d"] is not None and x["ret_5d"] > 0)
        with_5d = sum(1 for x in same if x["ret_5d"] is not None)
        avg_5d = round(sum(x["ret_5d"] for x in same if x["ret_5d"] is not None) / with_5d, 2) if with_5d else None
        stats[k] = {
            "n": len(same),
            "win_rate_5d": round(wins_5d / with_5d * 100, 1) if with_5d else None,
            "avg_5d": avg_5d,
        }

    result = {"entries": out_entries, "stats": stats, "asOf": str(end.date())}
    cache_set(cache_key, result)
    return result


# ============================================================================
# 投組 Drawdown 曲線
# ============================================================================
@app.get("/api/portfolio/drawdown-curve")
def api_portfolio_drawdown(days: int = 60):
    """每檔近 N 天從滾動高點 drawdown %。"""
    cache_key = f"pf_dd:{days}"
    cached = cache_get(cache_key)
    if cached:
        return cached
    p = load_portfolio()
    if not p:
        return {"holdings": [], "asOf": str(pd.Timestamp.today().date())}
    wl = load_watchlist()
    seen = {}
    for h in p:
        c = h.get("code")
        if c and c not in seen and c in wl:
            seen[c] = wl[c]["yf"]
    codes = list(seen.keys())
    yf_codes = [seen[c] for c in codes]
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=days + 10)
    try:
        data = yf.download(yf_codes, start=start, end=end, auto_adjust=False,
                           progress=False, group_by="ticker", threads=True)
    except Exception as e:
        raise HTTPException(503, f"yfinance fail: {e}")

    out_holdings = []
    for c, yfc in zip(codes, yf_codes):
        try:
            close = data[yfc]["Close"].dropna() if len(yf_codes) > 1 else data["Close"].dropna()
            if len(close) < 5: continue
            close = close.iloc[-days:]
            rolling_max = close.cummax()
            dd_pct = ((close - rolling_max) / rolling_max * 100)
            cur_dd = float(dd_pct.iloc[-1]); max_dd = float(dd_pct.min())
            series = [{"date": str(idx.date()), "price": round(float(p), 2),
                       "peak": round(float(rolling_max.iloc[i]), 2),
                       "dd_pct": round(float(dd_pct.iloc[i]), 2)}
                      for i, (idx, p) in enumerate(close.items())]
            out_holdings.append({
                "code": c, "name": wl.get(c, {}).get("name", c),
                "group": wl.get(c, {}).get("group", "—"),
                "cur_dd": round(cur_dd, 2), "max_dd": round(max_dd, 2),
                "current_price": round(float(close.iloc[-1]), 2),
                "peak_price": round(float(rolling_max.iloc[-1]), 2),
                "series": series[-30:],
            })
        except Exception:
            continue
    out_holdings.sort(key=lambda x: x["cur_dd"])
    result = {"holdings": out_holdings, "asOf": str(end.date()), "days": days}
    cache_set(cache_key, result)
    return result


# ============================================================================
# 投組再平衡建議
# ============================================================================
@app.get("/api/portfolio/rebalance")
def api_portfolio_rebalance(by: str = "group"):
    """依 group/code 目標權重算再平衡動作。"""
    target_cfg = load_json(REBALANCE_TARGET_FILE, {})
    p = load_portfolio()
    if not p:
        return {"holdings": [], "actions": [], "by": by, "total_value": 0,
                "warnings": ["無持股資料"]}
    wl = load_watchlist()
    holdings_value = {}
    holdings_info = {}
    total_value = 0.0
    for h in p:
        c = h.get("code")
        if not c: continue
        try:
            s = fetch_summary(c)
            price = float(s["price"])
        except Exception:
            price = float(h.get("cost_price", 0))
        shares = float(h.get("shares", 0))
        v = shares * price
        holdings_value[c] = holdings_value.get(c, 0) + v
        holdings_info[c] = {
            "name": wl.get(c, {}).get("name", c),
            "group": wl.get(c, {}).get("group", "—"),
            "price": price,
        }
        total_value += v
    if total_value <= 0:
        return {"holdings": [], "actions": [], "by": by, "total_value": 0,
                "warnings": ["持股總市值為 0"]}
    current = {}
    for c, v in holdings_value.items():
        if by == "code":
            current[c] = v
        else:
            g = holdings_info[c]["group"]
            current[g] = current.get(g, 0) + v
    target_pct = target_cfg.get(by, {})
    using_default = False
    if not target_pct:
        target_pct = {k: v / total_value * 100 for k, v in current.items()}
        using_default = True
    s = sum(target_pct.values())
    if s > 0 and abs(s - 100) > 0.5:
        target_pct = {k: v / s * 100 for k, v in target_pct.items()}
    actions = []
    for k in set(current.keys()) | set(target_pct.keys()):
        cur_v = current.get(k, 0)
        cur_pct = cur_v / total_value * 100
        tgt_pct = target_pct.get(k, 0)
        delta_pct = tgt_pct - cur_pct
        if abs(delta_pct) < 0.5:
            continue
        actions.append({
            "key": k, "current_pct": round(cur_pct, 2),
            "target_pct": round(tgt_pct, 2), "delta_pct": round(delta_pct, 2),
            "delta_value": round(total_value * delta_pct / 100, 2),
            "action": "買" if delta_pct > 0 else "賣",
        })
    actions.sort(key=lambda x: -abs(x["delta_pct"]))
    return {
        "by": by, "total_value": round(total_value, 2),
        "current": {k: round(v / total_value * 100, 2) for k, v in current.items()},
        "target": {k: round(v, 2) for k, v in target_pct.items()},
        "actions": actions, "using_default_target": using_default,
        "warnings": ["未設目標,目前用「維持現狀」為目標。可編輯 rebalance_target.json"]
                    if using_default else [],
        "asOf": str(pd.Timestamp.today().date()),
    }


@app.post("/api/portfolio/rebalance-target")
def api_set_rebalance_target(target: dict):
    cur = load_json(REBALANCE_TARGET_FILE, {})
    for k, v in (target or {}).items():
        cur[k] = v
    save_json(REBALANCE_TARGET_FILE, cur)
    return {"ok": True, "target": cur}


# ============================================================================
# 族群 alert 設定
# ============================================================================
@app.get("/api/group-alerts")
def api_get_group_alerts():
    return load_json(GROUP_ALERTS_FILE, {"groups": {}})


@app.post("/api/group-alerts")
def api_set_group_alerts(cfg: dict):
    """payload: {"groups": {"半導體設備": {"ret_1w_above": 5, "momentum_below": -3}}}"""
    existing = load_json(GROUP_ALERTS_FILE, {})
    existing["groups"] = cfg.get("groups", existing.get("groups", {}))
    save_json(GROUP_ALERTS_FILE, existing)
    return {"ok": True}


# ============================================================================
# 5) Fundamentals (月營收 + 季 EPS)
# ============================================================================
@app.get("/api/fundamentals/{code}")
def api_fundamentals(code: str):
    """從 FinMind 抓月營收 + 季 EPS。"""
    cached = cache_get(f"fund:{code}")
    if cached:
        return cached

    base = "https://api.finmindtrade.com/api/v4/data"
    start = (pd.Timestamp.today() - pd.Timedelta(days=900)).strftime("%Y-%m-%d")

    revenue = []
    try:
        r = requests.get(base, params={
            "dataset": "TaiwanStockMonthRevenue",
            "data_id": code,
            "start_date": start,
        }, timeout=8)
        data = r.json().get("data", [])
        if data:
            df = pd.DataFrame(data)
            df["ym"] = df["revenue_year"].astype(int).astype(str) + "/" + df["revenue_month"].astype(int).astype(str).str.zfill(2)
            df = df.sort_values(["revenue_year", "revenue_month"])
            # YoY: 本月對去年同月
            df["yoy"] = None
            for i in range(len(df)):
                this_y = int(df.iloc[i]["revenue_year"])
                this_m = int(df.iloc[i]["revenue_month"])
                last_yr = df[(df["revenue_year"] == this_y - 1) & (df["revenue_month"] == this_m)]
                if len(last_yr) and last_yr.iloc[0]["revenue"] > 0:
                    df.iloc[i, df.columns.get_loc("yoy")] = round((df.iloc[i]["revenue"] - last_yr.iloc[0]["revenue"]) / last_yr.iloc[0]["revenue"] * 100, 1)
            revenue = [{
                "ym":      r["ym"],
                "revenue": int(r["revenue"]) if r["revenue"] else 0,    # 千元
                "yoy":     float(r["yoy"]) if r["yoy"] is not None else None,
            } for r in df.tail(24).to_dict("records")]
    except Exception as e:
        print(f"[fund-rev] {code}: {e}")

    eps = []
    try:
        r = requests.get(base, params={
            "dataset": "TaiwanStockFinancialStatements",
            "data_id": code,
            "start_date": start,
        }, timeout=8)
        data = r.json().get("data", [])
        df = pd.DataFrame(data)
        if len(df):
            ed = df[df["type"] == "EPS"].sort_values("date")
            eps = [{
                "date":  d["date"],
                "value": float(d["value"]) if d["value"] is not None else 0,
            } for d in ed.tail(12).to_dict("records")]
    except Exception as e:
        print(f"[fund-eps] {code}: {e}")

    out = {"code": code, "revenue": revenue, "eps": eps}
    cache_set(f"fund:{code}", out)
    return out


# ============================================================================
# 6) TWII overlay (大盤連動)
# ============================================================================
@app.get("/api/index")
def api_index(period: str = "D"):
    """加權指數 ^TWII，回傳跟個股對齊用的 normalized 線。"""
    period = period.upper() if period.upper() in PERIOD_CFG else "D"
    cache_key = f"twii:{period}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    cfg = PERIOD_CFG[period]
    try:
        h = yf.Ticker("^TWII").history(period=cfg["period"], interval=cfg["interval"], auto_adjust=False)
    except Exception as e:
        return {"dates": [], "close": [], "error": str(e)}
    h = h.dropna(subset=["Close"]).iloc[-cfg["n"]:]
    if h.empty:
        return {"dates": [], "close": []}

    base = float(h["Close"].iloc[0])
    out = {
        "dates":   [idx.strftime("%m/%d") if period == "D" else idx.strftime("%Y/%m" if period == "M" else "%m/%d") for idx in h.index],
        "close":   [round(float(c), 2) for c in h["Close"].tolist()],
        "norm":    [round(float(c) / base * 100, 2) for c in h["Close"].tolist()],  # 起點 = 100
        "current": round(float(h["Close"].iloc[-1]), 2),
        "prev":    round(float(h["Close"].iloc[-2]), 2) if len(h) > 1 else round(base, 2),
    }
    cache_set(cache_key, out)
    return out


@app.get("/")
def root():
    return FileResponse(ROOT / "index.html")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, io
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    import uvicorn
    print("=" * 64)
    print("[TW Stock Intel v2]  http://localhost:18505/")
    print("    GET  /api/stocks                    清單摘要")
    print("    GET  /api/stock/{code}?period=D|W|M  詳細")
    print("    GET  /api/news/{code}                新聞")
    print("    GET  /api/groups                     族群")
    print("    GET  /api/ranking?by=change|volume|fi|rsi 熱度榜")
    print("    POST /api/watchlist                  新增")
    print("    DEL  /api/watchlist/{code}           移除")
    print("    POST /api/alerts/{code}              設警示")
    print("    POST /api/telegram                   設 bot")
    print("    GET  /api/refresh                    清快取")
    print("=" * 64)
    threading.Thread(target=alert_worker, daemon=True).start()
    uvicorn.run("server:app", host="0.0.0.0", port=18505, reload=False)
