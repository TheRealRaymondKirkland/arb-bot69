#!/bin/bash
cd /home/raykirkland88/arb-bot
while true; do
    if ! pgrep -f "python.*web_dashboard.py" > /dev/null; then
        echo "[$(date)] restarting bot..." >> /tmp/arb-bot-keepalive.log
        /home/raykirkland88/poly-trading-bot/venv/bin/python web_dashboard.py >> /tmp/arb-bot.log 2>&1 &
    fi
    sleep 30
done
