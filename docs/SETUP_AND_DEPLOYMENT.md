# Setup and Deployment Guide

Complete guide for setting up and deploying the YonEarth Gaia Chatbot.

---

## 📋 Prerequisites

### Required API Keys
```bash
OPENAI_API_KEY=sk-your-key-here          # Required for embeddings & LLM
PINECONE_API_KEY=your-pinecone-key-here  # Required for vector database
ELEVENLABS_API_KEY=your-key-here         # Optional for voice features
```

### System Requirements
- Python 3.9+
- Docker & Docker Compose (for containerized deployment)
- 2GB RAM minimum
- 10GB disk space

---

## 🚀 Quick Start (Docker)

### Option 1: One-Command Deployment
```bash
git clone https://github.com/DarrenZal/yonearth-gaia-chatbot.git
cd yonearth-gaia-chatbot
./deploy.sh
```

This automatically:
- ✅ Sets up Docker containers (nginx, Redis, API)
- ✅ Configures environment variables
- ✅ Starts the web interface
- ✅ Enables SSL (optional)

### Option 2: Manual Docker Compose
```bash
# 1. Clone repository
git clone https://github.com/DarrenZal/yonearth-gaia-chatbot.git
cd yonearth-gaia-chatbot

# 2. Create .env file
cp .env.example .env
nano .env  # Add your API keys

# 3. Start services
docker-compose up -d

# 4. View logs
docker-compose logs -f
```

**Access**: http://localhost:80

---

## 💻 Local Development Setup

### Install Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install requirements
pip install -r requirements.txt
```

### Start Development Server
```bash
# Quick start (recommended)
python scripts/start_local.py

# Or manual start
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Access**: http://localhost:8000

---

## 🖥️ VPS Production Deployment

### Server Setup (Ubuntu 20.04+)

#### 1. Install Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python, nginx, git
sudo apt install -y python3.9 python3-pip python3-venv nginx git

# Install Docker (optional)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

#### 2. Clone and Setup
```bash
# Clone to production directory
sudo git clone https://github.com/DarrenZal/yonearth-gaia-chatbot.git /root/yonearth-gaia-chatbot
cd /root/yonearth-gaia-chatbot

# Install Python dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create .env file
nano .env  # Add your API keys
```

#### 3. Configure Systemd Service
```bash
# Create service file
sudo nano /etc/systemd/system/yonearth-api.service
```

Add this configuration:
```ini
[Unit]
Description=YonEarth Gaia Chatbot API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/yonearth-gaia-chatbot
Environment="PATH=/root/yonearth-gaia-chatbot/venv/bin"
ExecStart=/root/yonearth-gaia-chatbot/venv/bin/uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable yonearth-api
sudo systemctl start yonearth-api
sudo systemctl status yonearth-api
```

#### 4. Configure Nginx
```bash
# Create nginx config
sudo nano /etc/nginx/sites-available/yonearth
```

Add this configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com;  # Or IP address

    # Serve static files
    location / {
        root /var/www/yonearth;
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # API docs
    location /docs {
        proxy_pass http://127.0.0.1:8000/docs;
    }
}
```

```bash
# Enable site and restart nginx
sudo ln -s /etc/nginx/sites-available/yonearth /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### 5. Deploy Frontend Files
```bash
# Create web directory
sudo mkdir -p /var/www/yonearth

# Copy web files
sudo cp /root/yonearth-gaia-chatbot/web/*.html /var/www/yonearth/
sudo cp /root/yonearth-gaia-chatbot/web/*.js /var/www/yonearth/
sudo cp /root/yonearth-gaia-chatbot/web/*.css /var/www/yonearth/

# Set permissions
sudo chown -R www-data:www-data /var/www/yonearth
```

---

## 🔄 Updating Production

### Development Directory Structure
```
/home/claudeuser/yonearth-gaia-chatbot/  # Edit files here
/root/yonearth-gaia-chatbot/             # Production API (systemd runs from here)
/var/www/yonearth/                       # Production web files (nginx serves from here)
```

### Update Backend (Python/API)
```bash
# 1. Copy updated Python files
sudo cp -r /home/claudeuser/yonearth-gaia-chatbot/src/* /root/yonearth-gaia-chatbot/src/

# 2. Restart API service
sudo systemctl stop yonearth-api
sleep 2
sudo systemctl start yonearth-api

# 3. Check status
sudo systemctl status yonearth-api
```

### Update Frontend (HTML/JS/CSS)
```bash
# 1. Copy updated web files
sudo cp /home/claudeuser/yonearth-gaia-chatbot/web/*.html /var/www/yonearth/
sudo cp /home/claudeuser/yonearth-gaia-chatbot/web/*.js /var/www/yonearth/
sudo cp /home/claudeuser/yonearth-gaia-chatbot/web/*.css /var/www/yonearth/

# 2. Update version numbers (cache-busting)
# In HTML files, increment version query parameters:
# <link rel="stylesheet" href="styles.css?v=2">  → v=3
# <script src="chat.js?v=2"></script>            → v=3

# 3. Reload nginx
sudo systemctl reload nginx
```

### Full Update (Both)
```bash
sudo cp -r /home/claudeuser/yonearth-gaia-chatbot/src/* /root/yonearth-gaia-chatbot/src/
sudo cp /home/claudeuser/yonearth-gaia-chatbot/web/* /var/www/yonearth/
sudo systemctl stop yonearth-api && sleep 2 && sudo systemctl start yonearth-api
sudo systemctl reload nginx
```

---

## 📊 Service Management

### Systemd Commands
```bash
# Check service status
sudo systemctl status yonearth-api

# Start service
sudo systemctl start yonearth-api

# Stop service
sudo systemctl stop yonearth-api

# Restart service
sudo systemctl restart yonearth-api

# Enable on boot (already enabled)
sudo systemctl enable yonearth-api

# View logs (follow mode)
sudo journalctl -u yonearth-api -f

# View recent logs
sudo journalctl -u yonearth-api --since "1 hour ago"
```

### Nginx Commands
```bash
# Test configuration
sudo nginx -t

# Reload (graceful restart)
sudo systemctl reload nginx

# Restart
sudo systemctl restart nginx

# Check status
sudo systemctl status nginx
```

---

## 🔍 Troubleshooting

### API Not Starting
```bash
# Check logs
sudo journalctl -u yonearth-api -n 50

# Check port availability
sudo netstat -tulpn | grep 8000

# Test manual start
cd /root/yonearth-gaia-chatbot
source venv/bin/activate
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

### Web Interface Not Loading
```bash
# Check nginx error logs
sudo tail -f /var/log/nginx/error.log

# Check file permissions
ls -la /var/www/yonearth/

# Test nginx config
sudo nginx -t
```

### Database Connection Issues
```bash
# Test Pinecone connection
python -c "from pinecone import Pinecone; pc = Pinecone(api_key='YOUR_KEY'); print(pc.list_indexes())"

# Check environment variables
cat /root/yonearth-gaia-chatbot/.env
```

---

## 🌐 SSL/HTTPS Setup (Optional)

### Using Let's Encrypt (Certbot)
```bash
# Install certbot
sudo apt install -y certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal (already configured)
sudo certbot renew --dry-run
```

---

## 📝 Environment Variables

### Required Variables
```bash
OPENAI_API_KEY=sk-...                    # OpenAI API key
PINECONE_API_KEY=...                     # Pinecone API key
PINECONE_INDEX_NAME=yonearth-episodes    # Pinecone index name
PINECONE_ENVIRONMENT=gcp-starter         # Pinecone environment
```

### Optional Variables
```bash
ELEVENLABS_API_KEY=...                   # For voice features
ELEVENLABS_VOICE_ID=...                  # Custom voice ID
OPENAI_MODEL=gpt-3.5-turbo              # LLM model
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
GAIA_PERSONALITY_VARIANT=warm_mother     # Default personality
GAIA_TEMPERATURE=0.7                     # LLM temperature
```

---

## ✅ Verification

### Test API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Test chat (original RAG)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "what is biochar?"}'

# Test BM25 search
curl -X POST http://localhost:8000/api/bm25/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "what is biochar?"}'
```

### Test Web Interface
1. Open browser: http://your-server-ip/
2. Send test message: "what is biochar?"
3. Verify response with episode citations
4. Check "Recommended Content" section

---

**Next Steps**: See [DEVELOPMENT.md](DEVELOPMENT.md) for development workflow and [ARCHITECTURE.md](ARCHITECTURE.md) for system details.
