import os
import json
import argparse
import requests
from typing import List, Dict, Any, Optional

def load_logs(log_file: str) -> List[Dict[str, Any]]:
    """Loads the step entries from a JSONL experiment log file."""
    steps = []
    
    if not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}")
        return steps
        
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    # The logger writes {event_type, data} wrappers
                    if entry.get('event_type') == 'step':
                        steps.append(entry['data'])
                    elif 'market_prices' in entry:
                        # Flat format (older runs)
                        steps.append(entry)
                except json.JSONDecodeError:
                    continue
    return steps

def get_polymarket_result(market_slug: str) -> Optional[str]:
    """Fetches the final resolution result for a market from Polymarket Gamma API."""
    try:
        url = "https://gamma-api.polymarket.com/public-search"
        resp = requests.get(url, params={"q": market_slug})
        resp.raise_for_status()
        data = resp.json()
        
        markets = []
        if isinstance(data, dict) and "events" in data:
            for event in data["events"]:
                markets.extend(event.get("markets", []))
        
        for m in markets:
            if m.get("slug") == market_slug:
                winner = m.get("winner")
                if winner and winner != "UNKNOWN":
                    return winner
        return None
    except Exception as e:
        print(f"Error fetching Polymarket result: {e}")
        return None

def evaluate_run(log_file: str):
    """Calculates ground truth verification metrics from a simulation run."""
    print(f"\n--- Evaluating Simulation: {log_file} ---")
    steps = load_logs(log_file)
    
    if not steps:
        print("No valid steps found to evaluate.")
        return

    # 1. Financial Performance (PnL / ROI)
    print("\n1. Financial Performance (PnL / ROI)")
    print("-" * 60)
    initial_value = steps[0].get("portfolio_value", 1000.0)
    final_value = steps[-1].get("portfolio_value", 1000.0)
    pnl = final_value - initial_value
    roi = (pnl / initial_value) * 100
    
    print(f"Starting Value: ${initial_value:.2f}")
    print(f"Final Value:    ${final_value:.2f}")
    print(f"Total PnL:      ${pnl:.2f} ({roi:+.2f}%)")

    # 2. Track Daily Belief vs True Price (Brier Score / MAE)
    total_error = 0.0
    total_squared_error = 0.0
    valid_points = 0
    
    print("\n2. Daily Price Alignment (Agent Belief vs. Market Truth)")
    print("-" * 60)
    for i, step in enumerate(steps):
        gt_data = step.get("ground_truth_verification", {})
        actual_prices = gt_data.get("actual_prices", {})
        agent_belief = gt_data.get("agent_belief")
        current_value = step.get("portfolio_value", 0.0)
        
        if actual_prices and agent_belief is not None:
            market_id = list(actual_prices.keys())[0]
            true_price = actual_prices[market_id]
            
            # Get action details
            action_data = step.get("action", {})
            action_type = action_data.get("action_type", action_data.get("action", "HOLD"))
            quantity = action_data.get("quantity", 0)
            
            # Get Portfolio State
            portfolio_obs = step.get("observation", {}).get("portfolio", {})
            cash_left = portfolio_obs.get("cash", 0.0)
            current_pos = portfolio_obs.get("positions", {}).get(market_id, 0)
            
            # Use logged execution price if available, else fallback to market price
            exec_price = gt_data.get("execution_price")
            if exec_price is None or exec_price == 0:
                exec_price = true_price
            
            error = abs(agent_belief - true_price)
            total_error += error
            total_squared_error += (agent_belief - true_price) ** 2
            valid_points += 1
            
            # Format action string: e.g. "BUY 100 @ 0.33" or "HOLD"
            if action_type != "HOLD" and quantity > 0:
                action_str = f"{action_type} {quantity} @ {exec_price:.2f}"
            else:
                action_str = "HOLD"
                
            print(f"Day {i+1:2d} | {action_str:15s} | Start-Cash: ${cash_left:7.2f} | Start-Pos: {current_pos:3d} | Price: {true_price:.2f} | Belief: {agent_belief:.2f} | End-Portfolio: ${current_value:7.2f}")

    if valid_points > 0:
        mae = total_error / valid_points
        brier = total_squared_error / valid_points
        print(f"\n=> Mean Absolute Error (MAE): {mae:.3f}")
        print(f"=> Brier Score (MSE):          {brier:.3f} (Lower is better)")
        
        if mae < 0.10:
            print("Verdict: Excellent Calibration.")
        elif mae < 0.25:
            print("Verdict: Acceptable Calibration.")
        else:
            print("Verdict: Poor Calibration.")

    # 3. Final Outcome Alignment
    print("\n3. Final Outcome Alignment")
    print("-" * 60)
    final_step = steps[-1]
    final_portfolio = final_step.get("observation", {}).get("portfolio", {})
    final_positions = final_portfolio.get("positions", {})
    
    gt_data = final_step.get("ground_truth_verification", {})
    final_prices = gt_data.get("actual_prices", {})
    
    if final_prices and final_positions:
        market_id = list(final_prices.keys())[0]
        final_price = final_prices[market_id]
        held_quantity = final_positions.get(market_id, 0)
        
        # 1. Try to get Winner from logged Metadata first (most reliable)
        metadata_winner = None
        for step in steps:
            if step.get("event_type") == "metadata":
                metadata_winner = step.get("data", {}).get("winner")
                break
        
        if metadata_winner:
            print(f"Logged Resolution (Metadata): {metadata_winner}")
            implied_truth = metadata_winner.upper()
        else:
            # 2. Try to get Archived Truth from API
            archived_winner = get_polymarket_result(market_id)
            if archived_winner and archived_winner != "UNKNOWN":
                print(f"Archived Resolution (API): {archived_winner}")
                implied_truth = archived_winner.upper() # "YES" or "NO"
            else:
                # 3. Fallback to Price Proxy (only if not at stalemate)
                if abs(final_price - 0.50) < 1e-6:
                    print(f"Predicted Outcome: STALEMATE (Price: {final_price:.2f})")
                    implied_truth = "UNKNOWN"
                else:
                    print(f"Predicted Outcome (Price Proxy): {final_price:.2f}")
                    implied_truth = "YES" if final_price > 0.5 else "NO"

        print(f"Final Agent Position: {held_quantity} shares")
        agent_bet = "YES" if held_quantity > 0 else ("NO" if held_quantity < 0 else "NEUTRAL")
        
        if implied_truth == "UNKNOWN":
            print("Verdict: INCONCLUSIVE. No clear ground truth available (Unresolved or Stale Data).")
        elif implied_truth == "YES" and agent_bet == "YES":
            print("Verdict: SUCCESS. Agent held positions for the winning outcome.")
        elif implied_truth == "NO" and agent_bet == "NO":
             print("Verdict: SUCCESS. Agent stayed away/shorted the losing outcome.")
        elif implied_truth == "NO" and agent_bet == "YES":
            print("Verdict: FAILED. Agent held positions for the losing outcome.")
        elif implied_truth == "YES" and agent_bet == "NO":
            print("Verdict: FAILED. Agent did not bet on the winning outcome.")
        else:
            print("Verdict: NEUTRAL/CAUTIOUS. Agent held no positions.")
            
    # 4. Rules & Reasoning
    print("\n4. Reasoning Excerpts")
    print("-" * 60)
    # Just show first, middle, and last to save space
    indices = [0, len(steps)//2, len(steps)-1]
    for idx in sorted(list(set(indices))):
        if idx < len(steps):
            action = steps[idx].get("action", {})
            reasoning = action.get("reasoning", "No reasoning.")
            print(f"Day {idx+1}: {reasoning[:200]}...")
        
    print("\nEvaluation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Research-Lookahead-AI Simulation Run")
    parser.add_argument("--log_file", type=str, required=True, help="Path to the experiment JSONL log file, e.g., logs/experiment_20260223_000133.jsonl")
    args = parser.parse_args()
    
    evaluate_run(args.log_file)
