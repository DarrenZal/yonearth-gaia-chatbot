# YonEarth Gaia Chatbot - VPS Deployment Guide

## 🚀 One-Command Deployment

This repository includes a complete Docker-based deployment system that can set up your YonEarth Gaia Chatbot on any VPS in minutes.

## ⚡ Quick Start

### Step 1: Clone and Run
```bash
# On your VPS (as root or user with sudo)
git clone https://github.com/DarrenZal/yonearth-gaia-chatbot.git
cd yonearth-gaia-chatbot
./deploy.sh
```

### Step 2: Add API Keys
The script will create a `.env` file. Edit it with your API keys:
```bash
nano .env
```

Required variables:
```bash
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
```

### Step 3: Restart
```bash
docker-compose restart
```

**That's it!** Your chatbot is live! 🎉

### Step 4: Access Your Chatbot
Your chatbot is now accessible at:
- **Web Interface**: `http://YOUR_SERVER_IP`
- **API Endpoint**: `http://YOUR_SERVER_IP/api/chat`

Replace `YOUR_SERVER_IP` with your actual server's public IP address.

## 🏗️ What Gets Deployed

### Services
- **🤖 YonEarth Gaia Chatbot**: FastAPI application with hybrid RAG
- **🗄️ Redis**: Caching for improved performance  
- **🌐 Nginx**: Reverse proxy serving web interface and API
- **🌍 Web Interface**: Beautiful chat interface accessible via browser
- **🔒 SSL/TLS**: Ready for Let's Encrypt certificates (optional)

### Features
- ✅ **Production-ready**: Multi-worker, health checks, auto-restart
- ✅ **Web Interface**: Complete chat UI accessible via IP address
- ✅ **API Access**: RESTful API endpoints for chat and recommendations
- ✅ **Secure**: Non-root containers, rate limiting, security headers, CORS
- ✅ **Scalable**: Redis caching, nginx load balancing ready
- ✅ **Monitored**: Health checks, logging, container management
- ✅ **SSL Ready**: Configuration ready for HTTPS setup

## 🔧 Manual Setup (Alternative)

If you prefer manual control:

### Prerequisites
```bash
# Install Docker & Docker Compose
sudo apt update
sudo apt install docker.io docker-compose git
sudo usermod -aG docker $USER
```

### Deploy
```bash
# Clone repository
git clone https://github.com/DarrenZal/yonearth-gaia-chatbot.git
cd yonearth-gaia-chatbot

# Configure environment
cp .env.example .env
nano .env  # Add your API keys

# Start services
docker-compose up -d
```

## 🌐 Domain & SSL Setup

### With Domain Name
1. **Point DNS**: Create an A record pointing to your VPS IP
2. **Run deployment**: The script will detect and configure your domain
3. **SSL Setup**: Choose 'y' when prompted for SSL certificates

### Domain Configuration
```bash
# Update nginx config
sed -i 's/localhost/your-domain.com/g' nginx.conf

# Update CORS origins
# Edit .env: ALLOWED_ORIGINS=https://your-domain.com,https://yonearth.org

# Restart
docker-compose restart
```

### Manual SSL Setup
```bash
# Get SSL certificate
docker-compose run --rm certbot certonly \
  --webroot \
  --webroot-path=/var/www/certbot/ \
  -d your-domain.com

# Update nginx config to enable HTTPS block
# Restart nginx
docker-compose restart nginx
```

## 🔍 Testing Your Deployment

### Web Interface Test
Open your browser and visit: `http://YOUR_SERVER_IP`

You should see:
- Beautiful green-themed chat interface
- "Chat with Gaia" title
- Welcome message from Gaia
- Working input field and send button

### API Health Check
```bash
curl http://YOUR_SERVER_IP/api/health
```

Note: Health endpoint may show an internal error, but this doesn't affect chat functionality.

### Test Chat API (The Accuracy Test!)
```bash
curl -X POST http://YOUR_SERVER_IP/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "what is biochar?",
    "max_results": 5,
    "session_id": "test"
  }'
```

**Expected**: Should reference Episodes 120, 122, and 165 (the correct ones!)

### Load Test
```bash
# Install ab (Apache Bench)
sudo apt install apache2-utils

# Test 100 requests, 10 concurrent
ab -n 100 -c 10 http://your-domain.com/health
```

## 📊 Management Commands

### Docker Deployment Management

#### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f app
docker-compose logs -f nginx
docker-compose logs -f redis
```

### Service Management
```bash
# Status
docker-compose ps

# Restart all
docker-compose restart

# Restart specific service
docker-compose restart app

# Stop all
docker-compose down

# Start all
docker-compose up -d

# Rebuild after code changes
docker-compose build app
docker-compose up -d app
```

### Resource Monitoring
```bash
# Docker stats
docker stats

# System resources
htop

# Disk usage
docker system df
```

### Alternative: Simple Server Management

For direct Python server deployment (without Docker):

#### Systemd Service Management (Recommended)

The server is now managed as a systemd service for better reliability:

##### Service Status & Control
```bash
# Check service status
sudo systemctl status yonearth-gaia

# Start the service
sudo systemctl start yonearth-gaia

# Stop the service
sudo systemctl stop yonearth-gaia

# Restart the service
sudo systemctl restart yonearth-gaia

# Enable auto-start on boot
sudo systemctl enable yonearth-gaia

# Disable auto-start on boot
sudo systemctl disable yonearth-gaia
```

##### View Logs
```bash
# Follow logs in real-time
sudo journalctl -u yonearth-gaia -f

# View logs from the last hour
sudo journalctl -u yonearth-gaia --since "1 hour ago"

# View last 100 lines
sudo journalctl -u yonearth-gaia -n 100

# Direct log file
tail -f /var/log/yonearth-gaia.log
```

##### Service Configuration
The service is configured at `/etc/systemd/system/yonearth-gaia.service`:
- Automatically restarts on failure
- Starts on system boot
- Logs to `/var/log/yonearth-gaia.log`
- Health monitoring via cron job every 5 minutes

##### Manual Management (Legacy)
```bash
# Start manually (not recommended)
python3 simple_server.py

# Start in background (not recommended - use systemd instead)
python3 simple_server.py &

# Check if running (manual process)
ps aux | grep simple_server

# Stop manual process
kill $(pgrep -f simple_server.py)

# Real-time system logs
journalctl -f | grep python
```

## 🔒 Security Considerations

### Firewall Setup
```bash
# Enable UFW
sudo ufw enable

# Allow SSH, HTTP, HTTPS
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Check status
sudo ufw status
```

### Security Features Included
- ✅ **Non-root containers**: App runs as 'app' user
- ✅ **Rate limiting**: 10 requests/minute per IP (configurable)
- ✅ **Security headers**: XSS protection, content type sniffing protection
- ✅ **CORS restrictions**: Limited to specified origins
- ✅ **Input validation**: Pydantic models validate all inputs
- ✅ **SSL/TLS**: HTTPS with strong cipher suites

### Additional Security (Recommended)
```bash
# Fail2Ban for SSH protection
sudo apt install fail2ban

# Automatic security updates
sudo apt install unattended-upgrades
sudo dpkg-reconfigure unattended-upgrades
```

## 🎛️ Configuration Options

### Environment Variables
```bash
# API Configuration
ALLOWED_ORIGINS=https://your-domain.com,https://yonearth.org
RATE_LIMIT_PER_MINUTE=20

# Performance Tuning
EPISODES_TO_PROCESS=172  # Process all episodes
CHUNK_SIZE=500          # Optimal for most queries
CHUNK_OVERLAP=50        # Good balance

# Gaia Personality
GAIA_PERSONALITY_VARIANT=warm_mother  # Options: warm_mother, wise_elder, playful_spirit
GAIA_TEMPERATURE=0.7                  # Creativity vs accuracy (0.0-1.0)
GAIA_MAX_TOKENS=1000                 # Response length
```

### Scaling Options
```bash
# More app workers (docker-compose.yml)
command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# More Redis memory
command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru

# Load balancing (add multiple app instances)
# See docker-compose.scale.yml for example
```

## 🔧 Troubleshooting

### Common Issues

**Port 80/443 already in use:**
```bash
sudo systemctl stop apache2 nginx
docker-compose up -d
```

**Out of disk space:**
```bash
# Clean Docker
docker system prune -a

# Clean logs
docker-compose logs --tail=100 app > recent.log
sudo rm /var/lib/docker/containers/*/*-json.log
```

**Memory issues:**
```bash
# Check memory usage
free -h
docker stats

# Reduce workers or episodes processed
# Edit .env: EPISODES_TO_PROCESS=50
```

**API key errors:**
```bash
# Check .env file
cat .env | grep API_KEY

# Restart after changes
docker-compose restart app
```

### Log Analysis
```bash
# Check startup logs
docker-compose logs app | head -50

# Check error logs
docker-compose logs app | grep ERROR

# Check nginx access logs
docker-compose exec nginx cat /var/log/nginx/access.log

# Check health endpoint
curl -v http://localhost:8000/health
```

## 📈 Performance Optimization

### Database Optimization
```bash
# Pinecone index optimization
# Ensure your index has correct dimensions (1536)
# Use appropriate similarity metric (cosine)
```

### Caching Strategy
```bash
# Redis optimization (redis.conf in container)
maxmemory 256mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### Resource Limits
```bash
# Add to docker-compose.yml services
deploy:
  resources:
    limits:
      cpus: '1.0'
      memory: 1G
    reservations:
      cpus: '0.5'
      memory: 512M
```

## 🔄 Updates & Maintenance

### Update Application
```bash
cd /opt/yonearth-chatbot
git pull origin main
docker-compose build app
docker-compose up -d app
```

### Update Dependencies
```bash
# Rebuild with latest packages
docker-compose build --no-cache app
docker-compose up -d app
```

### SSL Certificate Renewal
```bash
# Automatic renewal (cron job)
echo "0 12 * * * /usr/local/bin/docker-compose -f /opt/yonearth-chatbot/docker-compose.yml run --rm certbot renew --quiet" | sudo crontab -
```

### Backup Strategy
```bash
# Backup application data
docker-compose exec app tar -czf /app/backup.tar.gz /app/data

# Backup Redis data
docker-compose exec redis redis-cli BGSAVE

# Copy backups to host
docker cp yonearth-gaia-chatbot:/app/backup.tar.gz ./
```

## 💰 Cost Considerations

### VPS Requirements
- **Minimum**: 2GB RAM, 2 CPU cores, 20GB disk
- **Recommended**: 4GB RAM, 2 CPU cores, 40GB disk
- **Cost**: $10-20/month (DigitalOcean, Linode, Vultr)

### API Costs
- **OpenAI**: ~$0.002 per chat interaction
- **Pinecone**: Free tier (1M vectors)
- **Estimated**: $5-15/month for moderate usage

**Total Monthly Cost**: $15-35 for complete deployment

## 🎯 Production Checklist

Before going live:

- [ ] **API Keys**: Valid OpenAI and Pinecone keys configured
- [ ] **Domain**: DNS pointing to VPS IP address
- [ ] **SSL**: HTTPS certificates installed and working
- [ ] **Firewall**: UFW configured with appropriate rules
- [ ] **Backups**: Automated backup strategy in place
- [ ] **Monitoring**: Health checks and log monitoring set up
- [ ] **Testing**: Biochar query returns correct episodes (120, 122, 165)
- [ ] **Performance**: Load testing completed successfully
- [ ] **Security**: Security headers and rate limiting verified

## 🆘 Support

If you encounter issues:

1. **Check logs**: `docker-compose logs -f app`
2. **Verify health**: `curl http://localhost:8000/health`
3. **Test API keys**: Ensure keys are valid and have credits
4. **Check DNS**: Verify domain points to correct IP
5. **Restart services**: `docker-compose restart`

Your YonEarth Gaia Chatbot should now be running smoothly with hybrid search providing accurate, source-cited responses! 🌍✨