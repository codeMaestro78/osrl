#!/bin/bash
set -e
USER=$(whoami)
echo "[1/4] Installing systemd services for user: $USER"

sudo cp ~/os-rl-agent/deploy/osrl-collector.service \
        /etc/systemd/system/osrl-collector@${USER}.service
sudo cp ~/os-rl-agent/deploy/osrl-dashboard.service \
        /etc/systemd/system/osrl-dashboard@${USER}.service
sudo cp ~/os-rl-agent/deploy/osrl-auto-train.service \
        /etc/systemd/system/osrl-auto-train@${USER}.service

# Substitute %i with actual username in the unit files
sudo sed -i "s|%i|${USER}|g" \
  /etc/systemd/system/osrl-collector@${USER}.service \
  /etc/systemd/system/osrl-dashboard@${USER}.service \
  /etc/systemd/system/osrl-auto-train@${USER}.service

echo "[2/4] Reloading systemd daemon"
sudo systemctl daemon-reload

echo "[3/4] Enabling + starting services"
sudo systemctl enable --now osrl-collector@${USER}.service
sudo systemctl enable --now osrl-auto-train@${USER}.service
sudo systemctl enable --now osrl-dashboard@${USER}.service

echo "[4/4] Status:"
sudo systemctl status osrl-collector@${USER}.service --no-pager -l
sudo systemctl status osrl-auto-train@${USER}.service --no-pager -l
sudo systemctl status osrl-dashboard@${USER}.service --no-pager -l

echo ""
echo "✅ Done! Dashboard: http://localhost:8501"
echo ""
echo "Useful commands:"
echo "  journalctl -u osrl-collector@${USER} -f"
echo "  journalctl -u osrl-auto-train@${USER} -f"
echo "  journalctl -u osrl-dashboard@${USER} -f"
