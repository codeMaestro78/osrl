#!/bin/bash
# Quick public URL using Cloudflare Tunnel (no account needed for temp URLs)
# Install: curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared && chmod +x cloudflared
echo "Starting Cloudflare tunnel to http://localhost:8501 ..."
./cloudflared tunnel --url http://localhost:8501
