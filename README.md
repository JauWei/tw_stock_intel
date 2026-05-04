# 台股情報站 · TW Stock Intelligence Console

一頁式台股儀表板：即時行情、技術指標、籌碼面、訊號偵測、AI 評論、個人持股、Telegram 警示。

```
FastAPI + yfinance + FinMind + Google News + Gemini + ECharts
```

## 主要功能

- **即時資料**：yfinance 每日 OHLCV、FinMind 三大法人 10 日、Google News 中文 RSS
- **技術指標**：MA5/20/60、RSI(14)、MACD(12,26,9)、KD(9,3)、W底/M頭、ATR
- **訊號偵測**：黃金/死亡交叉、KD/RSI 超買賣、突破/跌破新高低（日/週/月自動換 lookback）
- **訊號回測**：點訊號徽章看歷史出現次數、勝率、後 5 天平均報酬、最佳最差
- **熱度榜**：6 種排序（漲幅/跌幅/量能/外資買超/外資賣超/RSI）
- **族群熱度地圖**：每族群平均漲幅、紅黑比、最強最弱
- **個人持股**：成本/市值/損益/報酬率/權重自動計算
- **AI 評論**：Gemini 2.0/1.5-flash 把指標 + 籌碼 + 訊號 + 持股餵進去產 4-6 句中文評估
- **Telegram 警示**：上下方價位設定，背景每 5 分鐘掃描，突破/跌破自動推播
- **多週期 K 線**：日 / 週 / 月切換 + 加權指數疊加
- **基本面**：FinMind 月營收 24 月（含 YoY）+ 季 EPS 12 季

## 啟動

需求：Python 3.9+

```bash
git clone https://github.com/JauWei/tw-stock-intel.git
cd tw-stock-intel
pip install -r requirements.txt
python server.py
```

開瀏覽器：[http://localhost:18505/](http://localhost:18505/)

### 也可以從 GitHub Pages 開（前端部份）

如果你把 repo 啟用 GitHub Pages，可以從 `https://你的帳號.github.io/tw-stock-intel/` 開前端 UI，**但本機仍需執行 `python server.py`**——前端會自動跨域連到 `localhost:18505`。

server 已開啟 CORS，從 GitHub Pages 載入的前端可以直接打本機 API。沒啟動 server 時 UI 會顯示「本機 server 未連線」。

### 📱 手機 / 平板開（與電腦同 WiFi）

**方法 A（推薦）**：手機開 `http://電腦LAN_IP:18505/`（如 `http://192.168.0.100:18505/`）。電腦 IP 從 `ipconfig` 找 IPv4。Windows 防火牆首次會跳對話框問允許私人網路，按允許。前端會自動偵測 host 帶 `:18505` 即為 server 同源，不必設定。

**方法 B**：手機開 GitHub Pages，工具列「⚙️ Server」填 `http://電腦IP:18505` → 儲存（存 localStorage）。⚠️ HTTPS 頁面連 HTTP server 可能被瀏覽器擋（Mixed Content），若不行請用方法 A。

## 設定（選用）

| 功能 | 怎麼設 | 取得 |
|---|---|---|
| Gemini AI 評論 | 工具列「🤖 AI」按鈕貼 API key | [Google AI Studio](https://aistudio.google.com/apikey) 免費 |
| Telegram 警示 | 工具列「🔔 Telegram」貼 bot_token + chat_id | Telegram 找 [@BotFather](https://t.me/BotFather) → /newbot |

兩個都不設也能完整使用儀表板，只是上述兩功能會 disabled。

## 資料夾結構

```
tw-stock-intel/
├── server.py              # FastAPI backend
├── index.html             # 單頁應用
├── requirements.txt
├── README.md
├── .gitignore
│
└── (執行後自動產生 ── 個人資料，已 ignore)
    ├── watchlist.json     # 觀察清單
    ├── portfolio.json     # 持股
    ├── alerts.json        # 警示閾值
    ├── telegram.json      # bot 設定
    └── gemini.json        # AI key
```

## API endpoints

```
GET  /                                  → index.html

# 觀察清單
GET    /api/stocks                      → 清單摘要 (含訊號)
GET    /api/stock/{code}?period=D|W|M   → 詳細
POST   /api/watchlist                   → {code, name?, tag?, group?}
DELETE /api/watchlist/{code}
GET    /api/probe/{code}                → 探測 .TW / .TWO

# 個股延伸
GET    /api/news/{code}                 → 新聞 (Google News 中文)
GET    /api/fundamentals/{code}         → 月營收 + 季 EPS
GET    /api/signal-stats/{code}/{key}   → 訊號歷史回測

# 視圖
GET    /api/groups                      → 族群分布
GET    /api/group-heatmap               → 族群熱度地圖
GET    /api/ranking?by=...              → 熱度榜 (change|down|volume|fi|fi_sell|rsi)

# 個人
GET    /api/portfolio                   → 持股 + 自動算市值/損益
POST   /api/portfolio                   → {code, shares, cost_price, ...}
DELETE /api/portfolio/{code}

# 警示 / 推播
POST   /api/telegram                    → {bot_token, chat_id}
GET    /api/telegram                    → configured 狀態
POST   /api/alerts/{code}               → {above, below}
DELETE /api/alerts/{code}
GET    /api/alerts

# AI
POST   /api/gemini                      → {api_key}
GET    /api/gemini                      → configured 狀態
GET    /api/ai-comment/{code}           → Gemini 中文評論

# 大盤
GET    /api/index?period=D|W|M          → 加權指數 ^TWII
GET    /api/refresh                     → 清快取
```

## 加自選股

在 UI 工具列點「➕ 新增股票」輸入代號，後端會自動探測 `.TW` / `.TWO` 並抓回 yfinance 名稱。

或直接編輯 [`server.py`](server.py) 的 `DEFAULT_WATCHLIST` 加入。

## 已知限制

- yfinance 對台股大約有 15 分鐘延遲，不適合當沖即時看盤
- FinMind 免費端點有速率限制（約 600 次/小時），16 檔以內足夠
- Telegram 警示需 server 持續運行（背景 thread 每 5 分鐘掃）
- AI 評論預設使用 `gemini-2.0-flash-exp`，API 變更時自動 fallback `gemini-1.5-flash`

## 授權

MIT — 歡迎 fork、修改、商用。

## Disclaimer

本工具為個人學習與資料整理用途，**不構成投資建議**，自負盈虧。
