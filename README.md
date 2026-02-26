# Research-Lookahead-AI: Sequential Multimodal Trading Agent

A high-fidelity research platform for evaluating Large Language Models (LLMs) and Vision-Language Models (VLMs) as sequential traders in historical prediction markets. This system simulates a real-world trading environment where agents must process streaming text, market data, and visual signals while managing a recursive "Journal" memory.

## 🌟 Key Features

*   **Multimodal Decision Making**: 
    *   **Visual Signals**: Agents analyze programmatically generated historical price charts (via `matplotlib`) and web images (capped at 5 per step) via the OpenAI Vision API.
    *   **Textual Context**: Processes rich news snippets from **Exa** and **Tavily** (Configurable content length).
*   **Time-Travel Simulation (Zero-Leakage Integrity)**:
    *   **External Cutoffs**: Implements strict `T-1s` news cutoff vs `T` market data, ensuring agents never see intraday "future" news.
    *   **Pre-training Bias Defense**: We cross-reference event dates against LLM knowledge cutoffs (e.g., GPT-4o's Oct 2023 cutoff) to ensure 100% "out-of-sample" testing.
    *   **Sliding Window Logic**: Fully dynamic news context (T-14, T-30, etc.) driven by the `--window` flag.
    *   **Temporal Guard**: A triple-layer defense (Native API filters + Query Engineering + Heuristic scans) catches streaming updates that might leak into historical results.
*   **Real Data Integration**:
    *   **Polymarket & Kalshi**: Robust providers for decentralized and regulated prediction markets.
    *   **Automated Charting**: Generates OHLC/Price charts on-the-fly for any historical window.
*   **Portfolio Management**: Tracks Cash, Positions, PnL, and enforces financial constraints (Starting $1000).
*   **Recursive Memory**: Uses a "Journaling" mechanism where the agent passes its evolving worldview, comparing signals against the **Market Resolution Rules**.
*   **Persistent Configuration**: Supports `.env` file for API key persistence.

## 📂 Project Structure

```
├── main.py                 # Core Simulation Entry point (ISO timestamp support)
├── evaluate.py             # Ground Truth Metrics (Brier, MAE, ROI) launcher [NEW]
├── .env.template           # Template for persistent API key configuration [NEW]
├── verify.py               # Mock verification loop
├── run_simulation.sh       # Interactive helper script
├── runs/                   # Unified output hierarchy [NEW]
│   └── [ticker]/[run_id]/  # Isolated folder per experiment
├── .env.template           # Template for persistent API key configuration
    ├── agents/
    │   ├── llm_agent.py    # Multi-turn logic & VLM handling
    │   ├── openai_provider.py # OpenAI Vision/Text wrapper (Base64 hardened)
    │   └── prompts.py      # Dynamic window-aware system templates
    ├── core/
    │   ├── environment.py  # Orchestration, Sub-second timing logic
    │   ├── portfolio.py    # Financial State Machine
    │   └── types.py        # Multimodal Action/Observation models
    └── data_loaders/
        ├── polymarket.py   # Real-time Volume, Rules, & CLOB data fetcher [UPDATED]
        ├── kalshi.py       # Regulated exchange provider
        └── context.py      # Multimodal News + Strict Temporal Guard [UPDATED]
```

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- API Keys: `OPENAI_API_KEY`, `EXA_API_KEY`, `TAVILY_API_KEY`.

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Running a High-Precision Simulation (main.py)
Use the following command to run a 14-day historical simulation with high-precision timing and a custom news window:

```bash
# Example: Sucre Mayoral Election
python3 main.py \
  --provider polymarket \
  --ticker "sucre-mayoral-election-winner-bolivia" \
  --question "Who will win the Sucre Mayoral Election in Bolivia?" \
  --start-date "2026-02-08T15:00:00" \
  --days 14 \
  --window 14 \
  --max-content 3000
```

### 📊 Output & Results
The system now uses a **Unified Run Directory** structure for better organization:
`runs/[ticker]/[question_slug]_[timestamp]/`
- `experiment.jsonl`: The machine-readable execution log (use for `evaluate.py`).
- `raw_data/`: Daily JSON snapshots containing exactly what the agent read and thought.
- `charts/`: Daily price charts (PNG) generated and analyzed by the agent.

To evaluate a specific run:
```bash
python3 evaluate.py --log_file runs/[ticker]/[run_id]/experiment.jsonl
```

### 📈 Evaluation Metrics
The evaluation script provides a deep-dive audit:
*   **Belief vs. Price**: Real-time calibration checking (how much the agent "trusts" its news vs. the market price).
*   **Brier Score**: Measures the accuracy of probabilistic predictions (lower is better).
*   **ROI/PnL**: Financial tracking including **Slippage** and **Bid/Ask Spreads**.
*   **Reasoning Audit**: Direct access to the agent's step-by-step logic for every trade.

## 🧠 Core Architecture: The "Temporal Guard"
To prevent **Look-ahead Bias**, the project implements a sophisticated filtering engine:
1.  **Native Filters**: Uses API-level `end_date` constraints for Exa and Tavily.
2.  **Query Modifiers**: Appends temporal context to search queries (e.g., "news before [Date]").
3.  **Heuristic Scan**: A secondary filter scans news headlines/content for "future outcome" keywords (e.g., "Defeated", "President-elect") to catch streaming updates that leak into historical results.

## 📊 Ground Truth Verification & Metrics
Detailed logs (JSONL) capture every multimodal signal and portfolio shift. The system validates against:
1.  **Price Calibration**: Brier Score and MAE via `evaluate.py`.
2.  **Outcome Alignment**: Final ROI vs. market resolution.
3.  **Rule Adherence**: Scraped from the Gamma API.
4.  **Volume Correlation**: Conviction check using 24h market volume.

View performance data in `logs/` and `raw_data/`.

## 🔮 Future Goals
1.  **Open Source VLMs**: Integrate LLaVA and Yi-VL via local providers.
2.  **Visual World Modeling**: Building latent representations of market "scenes" for predictive modeling.
3.  **Continual Learning**: Regret minimization over multi-week simulation runs.