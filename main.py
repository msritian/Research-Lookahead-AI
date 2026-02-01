import argparse
import os
import sys
from datetime import datetime, timedelta
from src.core.environment import MarketEnvironment
from src.agents.llm_agent import SequentialLLMAgent
from src.agents.openai_provider import OpenAIProvider
from src.agents.mock_provider import MockLLMProvider
from src.data_loaders.kalshi import KalshiDataProvider
from src.data_loaders.context import ContextDataProvider
from src.utils.logger import ExperimentLogger

def main():
    parser = argparse.ArgumentParser(description='Sequential Trader Simulation')
    parser.add_argument('--ticker', type=str, default="FED-RATE-CUT", help='Kalshi Market Ticker')
    parser.add_argument('--question', type=str, default="Will the Fed cut interest rates in March?", help='The specific question to predict')
    parser.add_argument('--start-date', type=str, default="2024-01-01", help='Start date YYYY-MM-DD')
    parser.add_argument('--days', type=int, default=14, help='Duration in days')
    parser.add_argument('--mock', action='store_true', help='Use mock LLM instead of OpenAI')
    
    args = parser.parse_args()

    # Parse Dates
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    except ValueError:
        print(f"Invalid date format: {args.start_date}")
        sys.exit(1)
        
    end_date = start_date + timedelta(days=args.days)
    
    print(f"--- Configuration ---")
    print(f"Market: {args.ticker}")
    print(f"Question: {args.question}")
    print(f"Period: {start_date.date()} to {end_date.date()} ({args.days} days)")
    print(f"Mode: {'MOCK' if args.mock else 'REAL (OpenAI)'}")
    print(f"---------------------")

    # 1. Initialize Data Providers
    # Kalshi
    kalshi_key = os.environ.get("KALSHI_API_KEY") # Optional for public data
    market_provider = KalshiDataProvider(api_key=kalshi_key)
    
    # Context (Exa/Tavily)
    # They check their own env vars
    context_provider = ContextDataProvider(
        sources=["exa", "tavily"],
        query_template=f"{{ticker}} {args.question} news" # Inject question into search
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
        
    agent = SequentialLLMAgent(llm_provider, market_question=args.question)
    
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
        market_ids=[args.ticker]
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
