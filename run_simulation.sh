#!/bin/bash

# run_simulation.sh - Helper to run the Sequential Trader Environment

echo "----------------------------------------------------------------"
echo "   Sequential Trader Evaluation Environment (Research-Lookahead-AI)"
echo "----------------------------------------------------------------"

# 1. Ask for Keys (if not set)
if [ -z "$OPENAI_API_KEY" ]; then
    echo -n "Enter OPENAI_API_KEY (leave blank to skip/mock): "
    read OPENAI_API_KEY
    if [ ! -z "$OPENAI_API_KEY" ]; then
        export OPENAI_API_KEY=$OPENAI_API_KEY
    fi
fi

if [ -z "$EXA_API_KEY" ]; then
    echo -n "Enter EXA_API_KEY (leave blank to skip): "
    read EXA_API_KEY
    if [ ! -z "$EXA_API_KEY" ]; then
        export EXA_API_KEY=$EXA_API_KEY
    fi
fi

if [ -z "$TAVILY_API_KEY" ]; then
    echo -n "Enter TAVILY_API_KEY (leave blank to skip): "
    read TAVILY_API_KEY
    if [ ! -z "$TAVILY_API_KEY" ]; then
        export TAVILY_API_KEY=$TAVILY_API_KEY
    fi
fi

# 2. Select Script
echo ""
echo "Select Simulation Mode:"
echo "1) Verify (Short loop, test mocks, hardcoded)"
echo "2) Full Simulation (Main)"
echo -n "Choice [1]: "
read CHOICE

if [ "$CHOICE" == "2" ]; then
    echo ""
    echo "--- Data Provider ---"
    echo "1) Polymarket (Public, no key needed)"
    echo "2) Kalshi (Requires API Key for history)"
    echo -n "Choice [1]: "
    read P_CHOICE
    
    PROVIDER="polymarket"
    if [ "$P_CHOICE" == "2" ]; then
        PROVIDER="kalshi"
    fi

    echo ""
    echo "--- Market Configuration ---"
    echo -n "Enter Market Query [Bitcoin]: "
    read TICKER
    TICKER=${TICKER:-Bitcoin}
    
    echo -n "Enter Specific Question [Will Bitcoin reach \$100k by 2025?]: "
    read QUESTION
    QUESTION=${QUESTION:-"Will Bitcoin reach \$100k by 2025?"}
    
    echo -n "Enter Start Date [2024-03-01]: "
    read START
    START=${START:-2024-03-01}
    
    echo -n "Enter Duration (Days) [14]: "
    read DAYS
    DAYS=${DAYS:-14}

    # Check mode
    FLAGS=""
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "No OpenAI Key detected. Forcing --mock mode."
        FLAGS="--mock"
    fi

    echo ""
    echo "Starting Simulation using $PROVIDER for $TICKER..."
    echo "----------------------------------------------------------------"
    python3 main.py --provider "$PROVIDER" --ticker "$TICKER" --question "$QUESTION" --start-date "$START" --days "$DAYS" $FLAGS
    
else
    echo ""
    echo "Starting Verification..."
    python3 verify.py
fi
