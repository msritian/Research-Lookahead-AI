import argparse
import os
import sys
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
    
    args = parser.parse_args()
    load_env()

    # Parse Dates (Supports YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
    try:
        if 'T' in args.start_date:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%dT%H:%M:%S")
        else:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    except ValueError:
        print(f"Invalid date format: {args.start_date}. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS")
        sys.exit(1)
        
    end_date = start_date + timedelta(days=args.days)
    
    context_window = args.window if args.window is not None else args.days

    print(f"--- Configuration ---")
    print(f"Provider: {args.provider}")
    print(f"Market: {args.ticker}")
    print(f"Question: {args.question}")
    print(f"Period: {start_date.date()} to {end_date.date()} ({args.days} days)")
    print(f"Context Window: {context_window} days")
    print(f"Max Content: {args.max_content} chars")
    print(f"Mode: {'MOCK' if args.mock else 'REAL (OpenAI)'}")
    print(f"---------------------")

    # 1. Initialize Data Providers
    if args.provider == "kalshi":
        kalshi_key = os.environ.get("KALSHI_API_KEY")
        market_provider = KalshiDataProvider(api_key=kalshi_key)
    else:
        market_provider = PolymarketDataProvider()
    
    # Context (Exa/Tavily)
    # They check their own env vars
    context_provider = ContextDataProvider(
        sources=["exa", "tavily"],
        query_template=f"{{ticker}} {args.question} news", # Inject question into search
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
        
    agent = SequentialLLMAgent(llm_provider, market_question=args.question, max_content=args.max_content)
    
    # 3. Initialize Logger
    logger = ExperimentLogger()
    
    # 4. Run Environment
    env = MarketEnvironment(
        start_date=start_date,
        end_date=end_date,
        market_provider=market_provider,
        context_provider=context_provider,
        agent=agent,
        logger=logger,
        market_ids=[args.ticker],
        context_window_days=context_window
    )
    
    print("Starting Simulation...")
    try:
        env.run()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    
    # Print Summary
    final_val = env.portfolio.get_state({}).total_value
    print(f"\nSimulation Complete.")
    print(f"Final Portfolio Value: ${final_val:.2f}")

if __name__ == "__main__":
    main()
