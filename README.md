# Research-Lookahead-AI: Sequential Multimodal Trading Agent

A high-fidelity research platform for evaluating Large Language Models (LLMs) and Vision-Language Models (VLMs) as sequential traders in historical prediction markets. This system simulates a real-world trading environment where agents must process streaming text, market data, and visual signals while managing a recursive "Journal" memory.

## 🌟 Key Features

*   **Multimodal Decision Making**: 
    *   **Visual Signals**: Agents analyze programmatically generated historical price charts (via `matplotlib`) and web images fetched from search results.
    *   **Textual Context**: Processes real-time news snippets from **Exa** and **Tavily**.
*   **Time-Travel Simulation (Anti-Cheating)**:
    *   **Temporal Integrity**: A triple-layer defense (Native API filters + Query Engineering + **Temporal Guard** heuristics) ensures agents cannot see "future" news during historical simulations.
    *   **Deterministic History**: Historical price data and charts are generated exactly as they would have appeared at the simulated timestamp.
*   **Real Data Integration**:
    *   **Polymarket & Kalshi**: Robust providers for decentralized and regulated prediction markets.
    *   **Automated Charting**: Generates OHLC/Price charts on-the-fly for any historical window.
*   **Portfolio Management**: Tracks Cash, Positions, PnL, and enforces financial constraints.
*   **Recursive Memory**: Uses a "Journaling" mechanism where the agent passes its evolving worldview from Day N to Day N+1.

## 📂 Project Structure

```
├── main.py                 # Core Simulation Entry point
├── verify.py               # Mock verification loop
├── run_simulation.sh       # Interactive helper script
├── charts/                 # Generated historical price charts
└── src/
    ├── agents/
    │   ├── llm_agent.py    # Multi-turn logic & VLM handling
    │   ├── openai_provider.py # OpenAI Vision/Text wrapper (Base64 hardened)
    │   └── prompts.py      # Multimodal system templates
    ├── core/
    │   ├── environment.py  # Orchestration, Time-loops, & Logging
    │   ├── portfolio.py    # Financial State Machine
    │   └── types.py        # Multimodal Action/Observation models
    └── data_loaders/
        ├── polymarket.py   # Historical Gamma/CLOB/Chart API [NEW]
        ├── kalshi.py       # Regulated exchange provider
        └── context.py      # Multimodal News + Temporal Guard [UPDATED]
```

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- API Keys: `OPENAI_API_KEY`, `EXA_API_KEY`, `TAVILY_API_KEY`.

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Running a Multimodal Simulation (Polymarket)
```bash
python3 main.py \
  --provider polymarket \
  --ticker "Will Donald Trump win the 2024 Election?" \
  --question "Will Donald Trump win the 2024 US Presidential Election?" \
  --start-date "2024-03-01" \
  --days 7
```

## 🧠 Core Architecture: The "Temporal Guard"
To prevent **Look-ahead Bias**, the project implements a sophisticated filtering engine:
1.  **Native Filters**: Uses API-level `end_date` constraints for Exa and Tavily.
2.  **Query Modifiers**: Appends temporal context to search queries (e.g., "news before [Date]").
3.  **Heuristic Scan**: A secondary filter scans news headlines/content for "future outcome" keywords (e.g., "Defeated", "President-elect") to catch streaming updates that leak into historical results.

## 📊 Evaluation & Logs
Detailed logs (JSONL) capture every multimodal signal, the agent's visual reasoning, and portfolio shifts. View them in `logs/`.

## 🔮 Future Goals
1.  **Open Source VLMs**: Integrate LLaVA and Yi-VL via local providers.
2.  **Visual World Modeling**: Building latent representations of market "scenes" for predictive modeling.
3.  **Continual Learning**: Regret minimization over multi-week simulation runs.