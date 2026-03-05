#!/bin/bash
# Deploy BullScore to Mac Mini
# Usage: ./deploy/install.sh

set -e

DEST="aj@100.71.249.98"
REMOTE_DIR="/Users/aj/bullscore"

echo "Creating remote directory..."
ssh "$DEST" "mkdir -p $REMOTE_DIR"

echo "Copying files..."
scp bull_score.py "$DEST:$REMOTE_DIR/"
scp test_bull_score.py "$DEST:$REMOTE_DIR/"

echo "Installing LaunchAgents..."
scp deploy/co.hyperday.bullscore.collect.plist "$DEST:~/Library/LaunchAgents/"
scp deploy/co.hyperday.bullscore.deliver.plist "$DEST:~/Library/LaunchAgents/"

echo "Loading LaunchAgents..."
ssh "$DEST" "launchctl unload ~/Library/LaunchAgents/co.hyperday.bullscore.collect.plist 2>/dev/null || true"
ssh "$DEST" "launchctl load ~/Library/LaunchAgents/co.hyperday.bullscore.collect.plist"
ssh "$DEST" "launchctl unload ~/Library/LaunchAgents/co.hyperday.bullscore.deliver.plist 2>/dev/null || true"
ssh "$DEST" "launchctl load ~/Library/LaunchAgents/co.hyperday.bullscore.deliver.plist"

echo "Running initial collect..."
ssh "$DEST" "python3 $REMOTE_DIR/bull_score.py --collect --no-cache" 2>&1

echo "Running self-test..."
ssh "$DEST" "python3 $REMOTE_DIR/bull_score.py --self-test" 2>&1

echo ""
echo "Done! BullScore deployed to Mac Mini."
echo "  Hourly collection: co.hyperday.bullscore.collect"
echo "  9am EET delivery:  co.hyperday.bullscore.deliver"
echo "  Logs: /tmp/bullscore-collect.log, /tmp/bullscore-deliver.log"
