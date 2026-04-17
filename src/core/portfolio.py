from typing import Dict
from .types import PortfolioState, TradeType, MarketSnapshot

class Portfolio:
    def __init__(self, initial_cash: float = 1000.0):
        self.cash = initial_cash
        self.positions: Dict[str, int] = {} # market_id -> quantity
        self.realized_pnl = 0.0
        self.initial_cash = initial_cash

    def execute_trade(self, market_id: str, trade_type: TradeType, quantity: int, price: float) -> bool:
        """
        Executes a trade. Returns True if successful, False if invalid (e.g. insufficient funds).
        Assumes 'price' is the execution price (e.g. Ask for Buy, Bid for Sell).
        """
        if quantity == 0:
            return True # No-op is always valid

        cost = quantity * price

        if trade_type == TradeType.BUY:
            if self.cash >= cost:
                self.cash -= cost
                self.positions[market_id] = self.positions.get(market_id, 0) + quantity
                return True
            else:
                return False # Insufficient funds state
        
        elif trade_type == TradeType.SELL:
            # Check if we have enough shares to sell
            current_holdings = self.positions.get(market_id, 0)
            if current_holdings >= quantity:
                revenue = quantity * price
                self.cash += revenue
                self.positions[market_id] = current_holdings - quantity
                # Update realized PnL logic could go here depending on accounting method (LIFO/FIFO/AvgCost)
                # For simplicity in this version, Realized PnP is updated at the end/settlement or implicitly tracked via Total Equity
                return True
            else:
                return False # Insufficient position
        
        return True

    def get_state(self, current_market_prices: Dict[str, float]) -> PortfolioState:
        """
        Returns the portfolio state marked to market.
        current_market_prices: Map of market_id -> current price (usually mid-point or last trade)
        """
        position_value = 0.0
        for m_id, qty in self.positions.items():
            # If price is missing, use 0 or last known. For now assume available.
            price = current_market_prices.get(m_id, 0.0)
            position_value += qty * price
        
        total_value = self.cash + position_value
        unrealized = total_value - self.initial_cash - self.realized_pnl # Simplified total return view

        return PortfolioState(
            cash=self.cash,
            positions=self.positions.copy(),
            unrealized_pnl=unrealized, # This is actually Total PnL in this simplified view
            realized_pnl=self.realized_pnl,
            total_value=total_value
        )
    
    def settle_market(self, market_id: str, settlement_price: float):
        """
        Handles market settlement (e.g., market resolves to YES=1.0 or NO=0.0).
        Converts positions to cash.
        """
        if market_id in self.positions:
            qty = self.positions[market_id]
            payout = qty * settlement_price
            self.cash += payout
            # This is a realization event
            # Cost basis logic is needed for accurate Realized vs Unrealized split
            # For this MVP, we just clear the position
            del self.positions[market_id]
