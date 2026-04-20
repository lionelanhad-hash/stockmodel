import base64
import io
import time
import warnings
import re
from datetime import datetime, timedelta

# Core Data Science
import numpy as np
import pandas as pd
import yfinance as yf
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Visualization & Web
from flask import Flask, request, render_template_string
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Sentiment & Analysis
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Add these specific Technical Analysis imports
try:
    from ta.trend import EMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("Warning: 'ta' library not found. Install via: pip install ta")


warnings.filterwarnings('ignore')

app = Flask(__name__)

# ==========================================
# 1. ASSET CONFIGURATION (The Brains)
# ==========================================
# This dictionary handles the specific rules for different markets
ASSET_CONFIG = {
    '^NSEI': {'name': 'Nifty 50', 'currency': '₹', 'market': 'IN', 'keywords': 'Nifty 50 OR Indian Stock Market OR Sensex'},
    '^BSESN': {'name': 'Sensex', 'currency': '₹', 'market': 'IN', 'keywords': 'Sensex OR BSE OR Indian Economy'},
    'RELIANCE.NS': {'name': 'Reliance Ind.', 'currency': '₹', 'market': 'IN', 'keywords': 'Reliance Industries OR Mukesh Ambani OR Jio'},
    'HDFCBANK.NS': {'name': 'HDFC Bank', 'currency': '₹', 'market': 'IN', 'keywords': 'HDFC Bank OR Indian Banking Sector'},
    'AAPL': {'name': 'Apple Inc.', 'currency': '$', 'market': 'US', 'keywords': 'Apple Inc OR iPhone OR Tech Stocks'},
    'NVDA': {'name': 'NVIDIA Corp.', 'currency': '$', 'market': 'US', 'keywords': 'NVIDIA OR AI Chips OR GPU Market'},
    'TSLA': {'name': 'Tesla Inc.', 'currency': '$', 'market': 'US', 'keywords': 'Tesla OR Elon Musk OR EV Market'},
    'BTC-USD': {'name': 'Bitcoin', 'currency': '$', 'market': 'CRYPTO', 'keywords': 'Bitcoin OR Cryptocurrency OR BTC'},
    'SHRIRAMFIN.NS': {'name': 'Shriram Finance', 'currency': '₹', 'market': 'IN', 'keywords': 'Shriram Finance OR NBFC OR Shriram Transport'},
    'INDUSTOWER.NS': {'name': 'Indus Towers', 'currency': '₹', 'market': 'IN', 'keywords': 'Indus Towers OR Bharti Infratel OR Telecom Towers India'},
    'SUZLON.NS': {
    'name': 'Suzlon Energy',
    'currency': '₹',
    'market': 'IN',
    'keywords': 'Suzlon Energy OR Suzlon Wind OR Wind Energy India OR Renewable Energy India'
},

}

def get_asset_info(symbol):
    return ASSET_CONFIG.get(symbol, {'name': symbol, 'currency': '$', 'market': 'US', 'keywords': 'Stock Market'})

# ==========================================
# 2. UI TEMPLATE (Dynamic Dashboard)
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ asset_name }} AI | Institutional Analytics</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            /* Fintech Dark Palette */
            --bg-body: #050505;
            --bg-panel: #0e0e0e;
            --bg-card: #141414;
            --border-subtle: #27272a;
            --border-active: #3f3f46;
            --text-main: #e4e4e7;
            --text-muted: #a1a1aa;
            --accent-primary: #3b82f6; 
            --accent-glow: rgba(59, 130, 246, 0.15);
            --signal-green: #10b981;
            --signal-red: #ef4444;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-body);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            background-image: radial-gradient(circle at 50% 0%, #1e1e24 0%, transparent 40%);
        }

        .container { width: 100%; max-width: 1400px; animation: fadeIn 0.6s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

        /* HEADER */
        .navbar {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 30px; padding: 20px 0; border-bottom: 1px solid var(--border-subtle);
        }
        .brand { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; letter-spacing: -1px; }
        .brand span { color: var(--accent-primary); }
        .status-pill { font-size: 0.75rem; padding: 6px 12px; border-radius: 100px; background: var(--border-subtle); border: 1px solid var(--border-active); }

        /* CONTROLS */
        .control-panel {
            background: var(--bg-card); border: 1px solid var(--border-subtle);
            border-radius: 12px; padding: 20px; margin-bottom: 25px;
            box-shadow: 0 20px 40px -10px rgba(0,0,0,0.5);
        }
        .form-row { display: flex; gap: 20px; align-items: flex-end; }
        .input-group { flex: 1; display: flex; flex-direction: column; gap: 8px; }
        .input-group label { font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); font-weight: 600; }
        
        .input-group input, .input-group select {
            background: var(--bg-panel); border: 1px solid var(--border-subtle);
            color: white; padding: 12px 16px; border-radius: 6px;
            font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; outline: none;
        }
        .input-group select { cursor: pointer; }
        .input-group input:focus, .input-group select:focus { border-color: var(--accent-primary); }

        .btn-predict {
            background: var(--accent-primary); color: white; border: none;
            padding: 12px 30px; border-radius: 6px; font-weight: 600; cursor: pointer;
            font-size: 0.95rem; height: 46px; transition: all 0.2s;
        }
        .btn-predict:hover { filter: brightness(1.1); }

        /* DASHBOARD */
        .dashboard { display: grid; grid-template-columns: 300px 1fr; gap: 25px; }
        @media (max-width: 900px) { 
            .dashboard { grid-template-columns: 1fr; } 
            .form-row { flex-direction: column; }
            .btn-predict { width: 100%; }
        }

        /* CARDS */
        .metric-card {
            background: var(--bg-card); border: 1px solid var(--border-subtle);
            border-radius: 10px; padding: 20px; position: relative; overflow: hidden; margin-bottom: 15px;
        }
        .metric-card::after { content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: var(--border-active); }
        .metric-card.bullish::after { background: var(--signal-green); }
        .metric-card.bearish::after { background: var(--signal-red); }
        .metric-card.primary::after { background: var(--accent-primary); }

        .metric-value { font-size: 1.75rem; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #fff; }
        .metric-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 8px; }
        .text-green { color: var(--signal-green); }
        .text-red { color: var(--signal-red); }

        /* CHART */
        .chart-panel {
            background: var(--bg-card); border: 1px solid var(--border-subtle);
            border-radius: 12px; padding: 20px; display: flex; flex-direction: column;
        }
        .chart-img-container {
            background: #000; border-radius: 8px; border: 1px solid var(--border-subtle);
            padding: 10px; display: flex; justify-content: center;
        }
        .chart-img-container img { max-width: 100%; height: auto; max-height: 450px; }

        /* TABLE */
        .table-container { margin-top: 25px; background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 12px; overflow: hidden; }
        table { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; }
        th { background: var(--bg-panel); text-align: left; padding: 15px 20px; color: var(--text-muted); text-transform: uppercase; font-size: 0.8rem; }
        td { padding: 15px 20px; border-bottom: 1px solid var(--border-subtle); color: var(--text-main); }
        .price-up { color: var(--signal-green); }

        /* LOADER */
        .loader-overlay {
            display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.9); z-index: 999; flex-direction: column; justify-content: center; align-items: center;
        }
        .loader-bar { width: 300px; height: 4px; background: var(--border-active); border-radius: 2px; overflow: hidden; margin-top: 20px; }
        .loader-progress { height: 100%; width: 0%; background: var(--accent-primary); animation: load 2s ease-in-out infinite; }
        @keyframes load { 0% { width: 0%; transform: translateX(-50%); } 100% { width: 100%; transform: translateX(0); } }
    </style>
    <script>
        function startLoading() { document.getElementById('loader').style.display = 'flex'; }
    </script>
</head>
<body>

    <div id="loader" class="loader-overlay">
        <div style="font-size: 1.5rem; font-weight: 700; letter-spacing: -1px;">CALIBRATING <span style="color:var(--accent-primary)">MODELS</span></div>
        <div class="loader-bar"><div class="loader-progress"></div></div>
    </div>

    <div class="container">
        <nav class="navbar">
            <div class="brand">{{ asset_name|upper }}<span>PRO</span>.AI</div>
            <div class="status-pill">● MARKET FEED ACTIVE</div>
        </nav>

        <div class="control-panel">
            <form action="/" method="post" onsubmit="startLoading()">
                <div class="form-row">
                    <div class="input-group">
                        <label>Asset Class</label>
                        <select name="symbol">
                            <option value="^NSEI" {% if symbol == '^NSEI' %}selected{% endif %}>Nifty 50 (India)</option>
                            <option value="^BSESN" {% if symbol == '^BSESN' %}selected{% endif %}>Sensex (India)</option>
                            <option value="RELIANCE.NS" {% if symbol == 'RELIANCE.NS' %}selected{% endif %}>Reliance Ind (India)</option>
                            <option value="HDFCBANK.NS" {% if symbol == 'HDFCBANK.NS' %}selected{% endif %}>HDFC Bank (India)</option>
                            <option value="AAPL" {% if symbol == 'AAPL' %}selected{% endif %}>Apple Inc (US)</option>
                            <option value="NVDA" {% if symbol == 'NVDA' %}selected{% endif %}>NVIDIA (US)</option>
                            <option value="TSLA" {% if symbol == 'TSLA' %}selected{% endif %}>Tesla (US)</option>
                            <option value="SHRIRAMFIN.NS" {% if symbol == 'SHRIRAMFIN.NS' %}selected{% endif %}>Shriram Finance (India)</option>
                            <option value="INDUSTOWER.NS" {% if symbol == 'INDUSTOWER.NS' %}selected{% endif %}>Indus Towers (India)</option>
                            <option value="SUZLON.NS" {% if symbol == 'SUZLON.NS' %}selected{% endif %}>
    Suzlon Energy (India)
</option>


                        </select>
                    </div>
                    <div class="input-group">
                        <label>Target Date</label>
                        <input type="date" name="date" value="{{ date }}" required>
                    </div>
                    <div class="input-group">
                        <label>Entry Time</label>
                        <input type="time" name="time" value="{{ time }}" required>
                    </div>
                    <div class="input-group">
                        <label>Horizon</label>
                        <select name="mode">
                            <option value="30" {% if mode == '30' %}selected{% endif %}>Intraday (30m Scalp)</option>
                            <option value="EOD" {% if mode == 'EOD' %}selected{% endif %}>Intraday (End of Day)</option>
                        </select>
                    </div>
                    <button type="submit" class="btn-predict">ANALYZE</button>
                </div>
            </form>
        </div>

        {% if error %}
        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid var(--signal-red); color: var(--signal-red); padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <strong>SYSTEM ERROR:</strong> {{ error }}
        </div>
        {% endif %}

        {% if plot_url %}
        <div class="dashboard">
            <div class="metrics-col">
                <div class="metric-card {% if sentiment_class == 'sentiment-pos' %}bullish{% elif sentiment_class == 'sentiment-neg' %}bearish{% else %}primary{% endif %}">
                    <div class="metric-label">Sentiment Signal</div>
                    <div class="metric-value {% if sentiment_class == 'sentiment-pos' %}text-green{% elif sentiment_class == 'sentiment-neg' %}text-red{% endif %}">
                        {{ sentiment_text }}
                    </div>
                    <div class="metric-sub">News Score: {{ sentiment_score }}</div>
                </div>

                <div class="metric-card primary">
                    <div class="metric-label">Forecast Avg Price</div>
                    <div class="metric-value">{{ currency }}{{ avg_price }}</div>
                </div>

                {% if metrics %}
                <div class="metric-card primary">
                    <div class="metric-label">Backtest Accuracy</div>
                    <div class="metric-value" style="color: var(--accent-primary)">{{ metrics.acc }}%</div>
                    <div class="metric-sub">MAE: {{ metrics.mae }}</div>
                </div>
                {% endif %}
            </div>

            <div class="chart-panel">
                <div class="chart-img-container">
                    <img src="data:image/png;base64,{{ plot_url }}" alt="Prediction Chart">
                </div>
            </div>
        </div>

        {% if predictions %}
        <div class="table-container">
             <table>
                <thead>
                    <tr>
                        <th width="30%">Timestamp</th>
                        <th width="40%">Predicted Close</th>
                        <th width="30%">Trend</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in predictions %}
                    <tr>
                        <td>{{ row.time }}</td>
                        <td class="price-up">{{ currency }}{{ row.price }}</td>
                        <td><span style="color: var(--text-muted);">PREDICTED</span></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        {% endif %}
    </div>
</body>
</html>
"""

# ==========================================
# 3. SENTIMENT ENGINE (Dynamic Keyword Support)
# ==========================================
class MarketSentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
    
    def clean_text(self, text):
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    def fetch_data(self, keywords):
        headlines = []
        try:
            api_key = "a2e3b4e6c2c147ad9308fd202b927fcd"
            # Use dynamic keywords from ASSET_CONFIG
            url = f"https://newsapi.org/v2/everything?q={keywords}&sortBy=publishedAt&language=en&apiKey={api_key}"
            resp = requests.get(url, timeout=3).json()
            headlines = [a['title'] for a in resp.get('articles', [])[:10]]
        except: pass

        reddit_texts = []
        try:
            # General financial subreddits cover most things
            headers = {'User-Agent': 'Mozilla/5.0'}
            url = "https://www.reddit.com/r/stocks/hot.json?limit=15"
            resp = requests.get(url, headers=headers, timeout=3).json()
            for post in resp['data']['children']:
                reddit_texts.append(post['data']['title'])
        except: pass
        
        return headlines + reddit_texts

    def analyze(self, keywords):
        texts = self.fetch_data(keywords)
        if not texts: return 0.0, 0.0

        scores = []
        for t in texts:
            clean = self.clean_text(t)
            vs = self.vader.polarity_scores(clean)['compound']
            tb = TextBlob(clean).sentiment.polarity
            scores.append((vs + tb) / 2)
        
        avg_score = np.mean(scores)
        return avg_score, np.std(scores)

# ==========================================
# 4. PREDICTION ENGINE (Dynamic Market Hours)
# ==========================================
class UniversalPredictor:
    def __init__(self, symbol):
        self.symbol = symbol
        self.info = get_asset_info(symbol)
        self.model = None
        self.feature_cols = None
        self.params = {
            'n_estimators': 800, 'learning_rate': 0.03, 'num_leaves': 40,
            'max_depth': 8, 'objective': 'regression', 'random_state': 42,
            'n_jobs': 1, 'verbose': -1
        }

    def fetch_data(self, date_str):
        # Adjust date if weekend
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        while dt.weekday() > 4: dt -= timedelta(days=1)
        
        start_date = dt - timedelta(days=365)
        end_date = dt + timedelta(days=1)
        
        # 1. Daily Data
        df_daily = yf.Ticker(self.symbol).history(start=start_date, end=end_date, interval="1d")
        
        # 2. Intraday Data
        df_intra = yf.Ticker(self.symbol).history(period="60d", interval="5m")
        
        if df_intra.empty: raise ValueError(f"No data found for {self.symbol}. Market might be closed or symbol invalid.")
        
        if df_intra.index.tz is not None: df_intra.index = df_intra.index.tz_localize(None)
        df_intra = df_intra.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'})
        
        # DYNAMIC MARKET FILTERING
        market = self.info['market']
        if market == 'IN':
            df_intra = df_intra.between_time('09:15', '15:30')
        elif market == 'US':
            df_intra = df_intra.between_time('09:30', '16:00')
        # Crypto (CRYPTO) has no time filter (24/7)

        return df_intra

    def calculate_vwap(self, df):
        """Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

    def detect_market_regime(self, df):
        """Identify volatility regime: 0=Low, 1=Normal, 2=High"""
        rolling_vol = df['log_ret'].rolling(20).std()
        # Use percentiles to determine regime dynamically
        vol_percentile = rolling_vol.rank(pct=True)
        
        regime = pd.Series(1, index=df.index) # Default Normal
        regime[vol_percentile < 0.33] = 0     # Low Vol
        regime[vol_percentile > 0.67] = 2     # High Vol
        return regime

    def engineer_advanced_features(self, df):
        """
        Generates 50+ Alpha Factors. 
        ROBUST VERSION: Uses fillna() instead of dropna() to prevent empty datasets.
        """
        df = df.copy()
        
        # === 1. BASIC TRANSFORMATIONS ===
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['co_ratio'] = (df['close'] - df['open']) / df['open']
        df['vol_change'] = df['volume'].pct_change()

        # === 2. LAGGED FEATURES ===
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'ret_lag_{lag}'] = df['log_ret'].shift(lag)
            df[f'vol_lag_{lag}'] = df['vol_change'].shift(lag)
            df[f'price_dist_{lag}'] = (df['close'] / df['close'].shift(lag)) - 1

        # === 3. ROLLING STATISTICS ===
        for window in [5, 10, 20]:
            df[f'ret_mean_{window}'] = df['log_ret'].rolling(window).mean()
            df[f'ret_std_{window}'] = df['log_ret'].rolling(window).std()
            df[f'vol_mean_{window}'] = df['volume'].rolling(window).mean()
            ma = df['close'].rolling(window).mean()
            df[f'ma_dist_{window}'] = (df['close'] - ma) / ma

        # === 4. ADVANCED STATISTICS ===
        df['ret_skew_20'] = df['log_ret'].rolling(20).skew()
        df['ret_kurt_20'] = df['log_ret'].rolling(20).kurt()
        df['autocorr_10'] = df['log_ret'].rolling(10).apply(lambda x: x.autocorr(lag=1) if len(x)>1 else 0, raw=False)

        # === 5. MICROSTRUCTURE ===
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        df['vwap_dist'] = (df['close'] - df['vwap']) / df['vwap']
        df['vol_price_corr'] = df['volume'].rolling(20).corr(df['close'])

        # === 6. TIME ENCODING ===
        minutes = df.index.hour * 60 + df.index.minute
        df['time_sin'] = np.sin(2 * np.pi * minutes / 1440)
        df['time_cos'] = np.cos(2 * np.pi * minutes / 1440)
        df['is_open'] = (minutes < (9*60 + 45)).astype(int) 
        df['is_close'] = (minutes > (15*60)).astype(int)

        # === 7. TECHNICAL INDICATORS ===
        if TA_AVAILABLE:
            try:
                df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
                bb = BollingerBands(df['close'], window=20)
                df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
                df['bb_pos'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
                df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            except Exception: pass

        # === 8. REGIME DETECTION ===
        rolling_vol = df['log_ret'].rolling(20).std()
        vol_rank = rolling_vol.rank(pct=True)
        df['regime'] = 1
        df.loc[vol_rank < 0.33, 'regime'] = 0
        df.loc[vol_rank > 0.67, 'regime'] = 2

        # === 9. CLEANING (THE FIX) ===
        # Instead of dropping, we fill forward, then backward, then 0.
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill().fillna(0)
        
        return df

    def train_and_predict(self, date_str, start_time, mode):
        # 1. Fetch Data
        raw_df = self.fetch_data(date_str)
        if raw_df.empty:
            raise ValueError(f"No data fetched for {self.symbol}. Market might be closed.")

        # 2. Engineer Features
        full_df = self.engineer_advanced_features(raw_df)
        
        # 3. Drop Non-Feature Columns
        drop_cols = ['open','high','low','close','volume','log_ret', 'vwap']
        X = full_df.drop(columns=[c for c in drop_cols if c in full_df.columns])
        y = full_df['log_ret'].shift(-1) # Target
        
        # 4. Filter Valid Rows
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X_all, y_all = X[valid_idx], y[valid_idx]

        split_point = int(len(X_all) * 0.8)
        X_train, X_test = X_all.iloc[:split_point], X_all.iloc[split_point:]
        y_train, y_test = y_all.iloc[:split_point], y_all.iloc[split_point:]
        
        if len(X_train) < 50:
            raise ValueError(f"Insufficient training data. Only {len(X_train)} rows available.")
        
        # 5. Train Model
        self.model = LGBMRegressor(**self.params)
        self.model.fit(X_train, y_train)
        self.feature_cols = X_train.columns.tolist()
        
        # 6. Prepare Simulation State
        target_dt = datetime.strptime(f"{date_str} {start_time}", "%Y-%m-%d %H:%M")
        
        if target_dt < raw_df.index[-1]:
            current_state_df = raw_df[raw_df.index <= target_dt].copy()
        else:
            current_state_df = raw_df.copy()

        if current_state_df.empty:
             raise ValueError(f"No data available before target time {target_dt}.")

        # 7. Generate Predictions
        # Recalculate features to ensure we have the latest state
        current_state_df = self.engineer_advanced_features(current_state_df)
        
        recent_volatility = current_state_df['log_ret'].tail(20).std()
        if np.isnan(recent_volatility) or recent_volatility == 0: recent_volatility = 0.0005 
            
        preds = []
        
        # === 🚀 DYNAMIC STEP CALCULATION ===
        if mode == '30':
            steps = 6  # Fixed 30 mins
        else:
            # Determine Close Time based on Market
            market = self.info['market']
            if market == 'IN':
                close_time_str = "15:30"
            elif market == 'US':
                close_time_str = "16:00"
            else:
                close_time_str = "23:59" # Crypto runs all day

            # Create full datetime object for market close on that specific date
            market_close_dt = datetime.strptime(f"{date_str} {close_time_str}", "%Y-%m-%d %H:%M")
            
            # Calculate time remaining
            time_diff = market_close_dt - target_dt
            
            # Convert to 5-minute chunks
            # total_seconds / 300 (which is 5 mins * 60 secs)
            calculated_steps = int(time_diff.total_seconds() / 300)
            
            # Safety: If user enters time AFTER market close, steps is 0. 
            # If Crypto, limit to reasonable max (e.g., 200)
            steps = max(0, calculated_steps)
            if market == 'CRYPTO': steps = min(steps, 288) # Cap at 24 hours

        current_price = current_state_df['close'].iloc[-1]
        
        for i in range(steps):
            # We must re-engineer at each step
            feat_df = self.engineer_advanced_features(current_state_df)
            
            last_row = feat_df[self.feature_cols].iloc[-1:].values
            pred_log_ret = self.model.predict(last_row)[0]
            
            noise = np.random.normal(0, recent_volatility)
            final_log_ret = pred_log_ret + noise
            
            next_price = current_price * np.exp(final_log_ret)
            
            # Calculate next timestamp
            next_time = current_state_df.index[-1] + timedelta(minutes=5)
            preds.append({'timestamp': next_time, 'price': next_price})
            
            # Simulate candle
            sim_high = next_price * (1 + (recent_volatility/2))
            sim_low = next_price * (1 - (recent_volatility/2))

            new_row = pd.DataFrame({
                'open': [next_price], 'high': [sim_high], 'low': [sim_low], 
                'close': [next_price], 'volume': [current_state_df['volume'].mean()],
                'log_ret': [final_log_ret] 
            }, index=[next_time])
            
            current_state_df = pd.concat([current_state_df, new_row])
            current_state_df['log_ret'] = current_state_df['log_ret'].fillna(0)
            current_price = next_price

        # --- 📊 METRICS CALCULATION (Add before the final return) ---
        
        # We use the training results to see how the model performed on known data
        test_preds = self.model.predict(X_test)

# 2. Update these variable names to match:
        dir_accuracy = np.mean(np.sign(test_preds) == np.sign(y_test))

        # 3. Use test_preds and y_test for PnL
        raw_returns = np.sign(test_preds) * y_test
        n_trades = len(test_preds)
        
        # Adjust these constants based on your broker (e.g., Zerodha/Interactive Brokers)
        slippage_pct = 0.0002   # 0.02% per trade
        brokerage_flat = 20.0   # Flat fee per trade
        est_trade_value = 100000 # Estimated position size for brokerage calc
        
        total_raw_return = raw_returns.sum()
        total_costs = (n_trades * slippage_pct) + (n_trades * (brokerage_flat / est_trade_value))
        net_pnl = total_raw_return - total_costs

        # 3. Sharpe Ratio (Annualized for 5-minute data)
        # 252 days * 75 five-minute intervals per day = 18,900
        if raw_returns.std() != 0:
            sharpe = np.sqrt(18900) * (raw_returns.mean() / raw_returns.std())
        else:
            sharpe = 0

        # --- 🖥️ TERMINAL OUTPUT ---
        print("\n" + "="*40)
        print(f" STRATEGY BACKTEST: {self.symbol} | {date_str}")
        print("="*40)
        print(f" Directional Accuracy: {dir_accuracy:.2%}")
        print(f" Net PnL (Post-Costs): {net_pnl:.4f}")
        print(f" Annualized Sharpe:    {sharpe:.2f}")
        print(f" Total Trades Evaluated: {n_trades}")
        print("="*40 + "\n")

        return pd.DataFrame(preds), raw_df

# ==========================================
# 5. MAIN ROUTE
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def home():
    # Defaults
    def_date = datetime.now().strftime('%Y-%m-%d')
    def_time = "10:00"
    def_symbol = "^NSEI"
    
    if request.method == 'GET':
        info = get_asset_info(def_symbol)
        return render_template_string(HTML_TEMPLATE, date=def_date, time=def_time, mode='30', 
                                      symbol=def_symbol, asset_name=info['name'], currency=info['currency'])

    try:
        # 1. Inputs
        date = request.form['date']
        time_input = request.form['time']
        mode = request.form['mode']
        symbol = request.form['symbol']
        
        # 2. Get Asset Info & Sentiment
        asset_info = get_asset_info(symbol)
        sent_analyzer = MarketSentimentAnalyzer()
        sent_score, sent_std = sent_analyzer.analyze(asset_info['keywords'])
        
        sent_class = "sentiment-pos" if sent_score > 0.05 else "sentiment-neg" if sent_score < -0.05 else "sentiment-neu"
        sent_text = "Bullish" if sent_score > 0.05 else "Bearish" if sent_score < -0.05 else "Neutral"

        # 3. Prediction
        predictor = UniversalPredictor(symbol)
        pred_df, hist_df = predictor.train_and_predict(date, time_input, mode)

        # 4. Accuracy (Backtest)
        metrics = None
        if not pred_df.empty:
            act_prices = []
            pred_prices = []
            for _, row in pred_df.iterrows():
                try:
                    act = hist_df.loc[row['timestamp']]['close']
                    act_prices.append(act)
                    pred_prices.append(row['price'])
                except KeyError: pass
            
            if len(act_prices) > 2:
                mae = mean_absolute_error(act_prices, pred_prices)
                mape = np.mean(np.abs(np.array(act_prices) - np.array(pred_prices)) / np.array(act_prices)) * 100
                metrics = {'mae': f"{mae:.2f}", 'acc': f"{100-mape:.2f}"}

        # 5. Plotting
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5))
        
        target_dt = datetime.strptime(f"{date} {time_input}", "%Y-%m-%d %H:%M")
        
        # --- A. Plot Past (Context) ---
        start_plot = target_dt - timedelta(minutes=120) # Show 2 hours before
        context_mask = (hist_df.index >= start_plot) & (hist_df.index <= target_dt)
        context_data = hist_df.loc[context_mask]
        
        if context_data.empty: context_data = hist_df.tail(30)
        ax.plot(context_data.index, context_data['close'], color='#555', label='Historical Context')
        
        # --- B. Plot Actual Future (The Truth) ---
        # We grab data AFTER the target time to compare with prediction
        future_end = target_dt + timedelta(minutes=35 if mode == '30' else 390)
        future_mask = (hist_df.index > target_dt) & (hist_df.index <= future_end)
        future_data = hist_df.loc[future_mask]
        
        if not future_data.empty:
            ax.plot(future_data.index, future_data['close'], color='white', linestyle=':', alpha=0.5, label='Actual Market Move')

        # --- C. Plot AI Prediction (The Forecast) ---
        if not pred_df.empty:
            ax.plot(pred_df['timestamp'], pred_df['price'], color='#3b82f6', linewidth=2.5, marker='o', markersize=4, label='AI Forecast')
            
            # Connect the prediction line to the historical line
            if not context_data.empty:
                ax.plot([context_data.index[-1], pred_df['timestamp'].iloc[0]], 
                        [context_data['close'].iloc[-1], pred_df['price'].iloc[0]], 
                        color='#3b82f6', linestyle='--')

        # Formatting
        ax.set_facecolor('#000')
        fig.patch.set_facecolor('#000')
        ax.grid(color='#333', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend(facecolor='#111', labelcolor='#ccc', edgecolor='#333')
        plt.tight_layout()

        # Save Plot
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)

        # 6. Formatting Output
        formatted_preds = []
        for _, row in pred_df.iterrows():
            formatted_preds.append({
                'time': row['timestamp'].strftime('%H:%M'),
                'price': f"{row['price']:.2f}"
            })

        return render_template_string(HTML_TEMPLATE,
                                      date=date, time=time_input, mode=mode, symbol=symbol,
                                      asset_name=asset_info['name'], currency=asset_info['currency'],
                                      plot_url=plot_url,
                                      predictions=formatted_preds,
                                      sentiment_score=f"{sent_score:.3f}",
                                      sentiment_class=sent_class,
                                      sentiment_text=sent_text,
                                      avg_price=f"{pred_df['price'].mean():.2f}",
                                      metrics=metrics)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template_string(HTML_TEMPLATE, date=request.form.get('date'), time=request.form.get('time'), error=str(e), symbol="^NSEI", asset_name="Error", currency="")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
