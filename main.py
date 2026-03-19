import argparse
import os
import sys
import re
from datetime import datetime, timedelta
from src.core.environment import MarketEnvironment
from src.agents.llm_agent import SequentialLLMAgent
from src.agents.openai_provider import OpenAIProvider
from src.agents.mock_provider import MockLLMProvider
from src.data_loaders.kalshi import KalshiDataProvider
from src.data_loaders.polymarket import PolymarketDataProvider
from src.data_loaders.context import ContextDataProvider
from src.utils.logger import ExperimentLogger

def load_env():
    """Simple manual .env loader to avoid extra dependencies."""
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip('"').strip("'")
        print(f"Loaded environment variables from {env_path}")

def slugify(text: str) -> str:
    """Converts a string to a safe filesystem-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    text = re.sub(r'^-+|-+$', '', text)
    return text

def run_simulation(args, market_ticker, market_question, start_date, context_window, run_dir, metadata=None):
    """Orchestrates a single simulation run."""
    print(f"\n--- Initializing Simulation: {market_ticker} ---")
    end_date = start_date + timedelta(days=args.days)
    
    # 1. Initialize Data Providers
    if args.provider == "kalshi":
        kalshi_key = os.environ.get("KALSHI_API_KEY")
        market_provider = KalshiDataProvider(api_key=kalshi_key)
    else:
        market_provider = PolymarketDataProvider()
    
    # Context (Exa/Tavily)
    context_provider = ContextDataProvider(
        sources=["exa", "tavily"],
        query_template=f"{{ticker}} {market_question} news",
        max_content=args.max_content
    )
    
    # 2. Initialize Agent
    if args.mock:
        llm_provider = MockLLMProvider()
    else:
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            print("Error: OPENAI_API_KEY not found. Set it or use --mock.")
            sys.exit(1)
        llm_provider = OpenAIProvider(api_key=openai_key)
        
    agent = SequentialLLMAgent(llm_provider, market_question=market_question, max_content=args.max_content)
    
    # 3. Initialize Logger
    logger = ExperimentLogger(run_dir=run_dir, metadata=metadata)
    
    # 4. Run Environment
    env = MarketEnvironment(
        start_date=start_date,
        end_date=end_date,
        market_provider=market_provider,
        context_provider=context_provider,
        agent=agent,
        logger=logger,
        market_ids=[market_ticker],
        context_window_days=context_window,
        run_dir=run_dir
    )
    
    print(f"Starting Simulation Period: {start_date.date()} to {end_date.date()}")
    try:
        env.run()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        return None
    
    return env.portfolio.get_state({}).total_value

def main():
    parser = argparse.ArgumentParser(description='Sequential Trader Simulation')
    parser.add_argument('--ticker', type=str, default="Bitcoin", help='Market Ticker or Search Query')
    parser.add_argument('--question', type=str, default="Will Bitcoin reach $100k by 2025?", help='The specific question to predict')
    parser.add_argument('--start-date', type=str, default="2024-03-01", help='Start date YYYY-MM-DD')
    parser.add_argument('--days', type=int, default=14, help='Duration in days to run the simulation')
    parser.add_argument('--window', type=int, default=None, help='Context window in days for news/data (defaults to same as --days)')
    parser.add_argument('--max-content', type=int, default=2000, help='Maximum characters per news article content')
    parser.add_argument('--mock', action='store_true', help='Use mock LLM instead of OpenAI')
    parser.add_argument('--provider', type=str, default="polymarket", choices=["kalshi", "polymarket"], help='Data provider to use')
    
    # Hindsight Options
    parser.add_argument('--hindsight-query', type=str, help='Search for archived/closed markets by keyword')
    parser.add_argument('--hindsight-limit', type=int, default=3, help='Max number of archived markets to simulate')
    parser.add_argument('--sort-latest', action='store_true', help='Sort hindsight results by latest end date')

    args = parser.parse_args()
    load_env()

    # Determine Simulation Targets
    targets = []
    if args.hindsight_query:
        print(f"Hindsight Engine: Discovering historical markets for '{args.hindsight_query}'...")
        discovery_provider = PolymarketDataProvider() if args.provider == "polymarket" else KalshiDataProvider()
        found_markets = discovery_provider.discover_markets(
            args.hindsight_query, 
            limit=args.hindsight_limit,
            sort_latest=args.sort_latest
        )
        
        if not found_markets:
            print("No archived markets found for query.")
            sys.exit(0)
            
        for m in found_markets:
            # Parse endDate: can be "2024-07-26" or ISO
            end_val = m['end_date']
            try:
                if 'T' in end_val:
                    end_dt = datetime.fromisoformat(end_val.replace('Z', '+00:00')).replace(tzinfo=None)
                else:
                    end_dt = datetime.strptime(end_val, "%Y-%m-%d")
            except:
                print(f"Warning: Could not parse end date {end_val} for {m['ticker']}, skipping.")
                continue

            start_dt = end_dt - timedelta(days=args.days)
            targets.append({
                "ticker": m['ticker'],
                "question": m['question'],
                "start_date": start_dt,
                "metadata": {
                    "winner": m.get('winner'),
                    "rules": m.get('rules'),
                    "closed": m.get('closed')
                }
            })
    else:
        # Standard Single Targeted Target
        try:
            if 'T' in args.start_date:
                start_dt = datetime.strptime(args.start_date, "%Y-%m-%dT%H:%M:%S")
            else:
                start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            print(f"Invalid date format: {args.start_date}. Use YYYY-MM-DD")
            sys.exit(1)
            
        targets.append({
            "ticker": args.ticker,
            "question": args.question,
            "start_date": start_dt
        })

    print(f"Found {len(targets)} targets for simulation.")

    # Loop through targets
    for target in targets:
        context_window = args.window if args.window is not None else args.days
        
        # Generate Run Directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ticker_slug = slugify(target['ticker'])
        question_slug = slugify(target['question'])[:50]
        run_id = f"{question_slug}_{timestamp}"
        run_dir = os.path.join("runs", ticker_slug, run_id)
        
        final_val = run_simulation(
            args=args,
            market_ticker=target['ticker'],
            market_question=target['question'],
            start_date=target['start_date'],
            context_window=context_window,
            run_dir=run_dir,
            metadata=target.get('metadata')
        )
        
        if final_val is not None:
            print(f"\n--- Simulation Complete ---")
            print(f"Ticker: {target['ticker']}")
            print(f"Final Value: ${final_val:.2f}")
            print(f"Log: {os.path.join(run_dir, 'experiment.jsonl')}")
            print(f"---------------------------\n")

if __name__ == "__main__":
    main()
