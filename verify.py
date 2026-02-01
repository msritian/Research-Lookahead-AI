from datetime import datetime, timedelta
from src.core.environment import MarketEnvironment
from src.agents.llm_agent import SequentialLLMAgent
from src.agents.mock_provider import MockLLMProvider
from src.data_loaders.kalshi import KalshiDataProvider
from src.data_loaders.context import ContextDataProvider
from src.utils.logger import ExperimentLogger

def main():
    # Setup
    start_date = datetime(2024, 3, 1)
    end_date = start_date + timedelta(days=2) # 2 days
    
    # 1. Market Provider (Real wrapper, but will return placeholder if API fails)
    market_provider = KalshiDataProvider(api_key="mock_key")
    
    # 2. Context Provider (Flexible Aggregator)
    # in 'real' usage, we'd pass api keys or they load from env
    # This will use the Mock logic inside ExaSource/TavilySource because no keys are in env
    context_provider = ContextDataProvider(
        sources=["exa", "tavily"],
        query_template="{ticker} latest context"
    )
    
    # 3. Agent
    llm_provider = MockLLMProvider()
    agent = SequentialLLMAgent(llm_provider)
    
    logger = ExperimentLogger(log_dir="logs_verification")
    
    env = MarketEnvironment(
        start_date=start_date,
        end_date=end_date,
        market_provider=market_provider,
        context_provider=context_provider,
        agent=agent,
        logger=logger,
        market_ids=["FED-RATE-CUT"]
    )
    
    print("Running Verification Simulation...")
    # This will print "Error fetching..." for API calls but should proceed with fallbacks
    env.run()
    print("Verification Complete. Check logs.")

if __name__ == "__main__":
    main()
