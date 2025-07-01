#!/bin/bash

# YonEarth Gaia Chatbot - VPS Deployment Script
# This script sets up the complete Docker-based deployment

set -e  # Exit on any error

echo "🚀 YonEarth Gaia Chatbot - VPS Deployment Script"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   log_warning "This script is running as root. Consider using a non-root user for better security."
fi

# Step 1: Update system and install Docker
log_info "Step 1: Installing Docker and Docker Compose..."

if ! command -v docker &> /dev/null; then
    log_info "Installing Docker..."
    
    # Update package list
    sudo apt-get update
    
    # Install required packages
    sudo apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # Add Docker repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker Engine
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    
    # Add current user to docker group
    sudo usermod -aG docker $USER
    
    log_success "Docker installed successfully!"
else
    log_success "Docker already installed"
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    log_info "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    log_success "Docker Compose installed successfully!"
else
    log_success "Docker Compose already installed"
fi

# Step 2: Clone repository (if not already cloned)
log_info "Step 2: Setting up application directory..."

if [ ! -d "/opt/yonearth-chatbot" ]; then
    log_info "Cloning repository..."
    sudo mkdir -p /opt/yonearth-chatbot
    sudo chown $USER:$USER /opt/yonearth-chatbot
    cd /opt/yonearth-chatbot
    git clone https://github.com/DarrenZal/yonearth-gaia-chatbot.git .
    log_success "Repository cloned successfully!"
else
    log_info "Repository already exists. Pulling latest changes..."
    cd /opt/yonearth-chatbot
    git pull origin main
fi

# Step 3: Configure environment
log_info "Step 3: Setting up environment configuration..."

if [ ! -f ".env" ]; then
    log_info "Creating .env file from template..."
    cp .env.example .env
    
    log_warning "⚠️  IMPORTANT: You need to edit .env file with your API keys!"
    echo
    echo "Required API keys:"
    echo "1. OPENAI_API_KEY - Get from https://platform.openai.com/api-keys"
    echo "2. PINECONE_API_KEY - Get from https://app.pinecone.io/"
    echo
    echo "Optional configurations:"
    echo "3. ALLOWED_ORIGINS - Add your domain"
    echo "4. Other settings as needed"
    echo
    
    read -p "Press Enter to open .env file for editing (or edit manually later)..."
    nano .env || vim .env || echo "Please edit .env file manually"
else
    log_success ".env file already exists"
fi

# Step 4: Configure domain (optional)
log_info "Step 4: Domain configuration..."

read -p "Do you have a domain name for this deployment? (y/n): " has_domain

if [ "$has_domain" = "y" ] || [ "$has_domain" = "Y" ]; then
    read -p "Enter your domain name (e.g., chat.yonearth.org): " domain_name
    
    if [ ! -z "$domain_name" ]; then
        log_info "Updating nginx configuration for domain: $domain_name"
        sed -i "s/localhost/$domain_name/g" nginx.conf
        
        # Update allowed origins in .env
        sed -i "s|ALLOWED_ORIGINS=.*|ALLOWED_ORIGINS=https://$domain_name,https://yonearth.org|g" .env
        
        log_success "Domain configuration updated!"
        
        # Ask about SSL
        read -p "Do you want to set up SSL certificates? (y/n): " setup_ssl
        if [ "$setup_ssl" = "y" ] || [ "$setup_ssl" = "Y" ]; then
            log_info "SSL will be configured after initial deployment"
        fi
    fi
fi

# Step 5: Build and start services
log_info "Step 5: Building and starting services..."

# Stop any existing containers
docker-compose down 2>/dev/null || true

# Build and start services
log_info "Building Docker images..."
docker-compose build

log_info "Starting services..."
docker-compose up -d

# Wait for services to be ready
log_info "Waiting for services to start..."
sleep 30

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    log_success "Services started successfully!"
    
    # Test the health endpoint
    log_info "Testing health endpoint..."
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "Health check passed! ✅"
    else
        log_warning "Health check failed. Check logs with: docker-compose logs app"
    fi
else
    log_error "Some services failed to start. Check logs with: docker-compose logs"
    exit 1
fi

# Step 6: SSL setup (if domain was provided)
if [ ! -z "$domain_name" ] && [ "$setup_ssl" = "y" ]; then
    log_info "Step 6: Setting up SSL certificates..."
    
    # Stop nginx temporarily
    docker-compose stop nginx
    
    # Get SSL certificate
    docker-compose run --rm certbot certonly \
        --webroot \
        --webroot-path=/var/www/certbot/ \
        -d $domain_name \
        --email admin@$domain_name \
        --agree-tos \
        --no-eff-email
    
    if [ $? -eq 0 ]; then
        log_success "SSL certificate obtained!"
        
        # Update nginx config to enable HTTPS
        log_info "Enabling HTTPS in nginx configuration..."
        sed -i 's|server_name localhost|server_name '$domain_name'|g' nginx.conf
        sed -i 's|# return 301 https|return 301 https|g' nginx.conf
        
        # Uncomment HTTPS server block
        sed -i '/# server {/,/# }/s/^#[[:space:]]*//' nginx.conf
        sed -i 's|your-domain.com|'$domain_name'|g' nginx.conf
        
        # Restart nginx with new config
        docker-compose up -d nginx
        
        log_success "HTTPS enabled! Your chatbot is available at https://$domain_name"
    else
        log_error "Failed to obtain SSL certificate"
        docker-compose up -d nginx
    fi
fi

# Step 7: Final setup and information
log_info "Step 7: Final setup..."

# Create systemd service for auto-restart
if [ ! -f "/etc/systemd/system/yonearth-chatbot.service" ]; then
    log_info "Creating systemd service for auto-restart..."
    
    sudo tee /etc/systemd/system/yonearth-chatbot.service > /dev/null <<EOF
[Unit]
Description=YonEarth Gaia Chatbot
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/yonearth-chatbot
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl enable yonearth-chatbot
    log_success "Systemd service created and enabled!"
fi

# Display final information
echo
echo "🎉 Deployment Complete!"
echo "======================"
echo
log_success "Your YonEarth Gaia Chatbot is now running!"
echo
echo "📍 Access Information:"
if [ ! -z "$domain_name" ]; then
    if [ "$setup_ssl" = "y" ]; then
        echo "   🌐 Web URL: https://$domain_name"
        echo "   🔍 Health Check: https://$domain_name/health"
        echo "   💬 Chat API: https://$domain_name/chat"
    else
        echo "   🌐 Web URL: http://$domain_name"
        echo "   🔍 Health Check: http://$domain_name/health"
        echo "   💬 Chat API: http://$domain_name/chat"
    fi
else
    echo "   🌐 Local URL: http://localhost"
    echo "   🔍 Health Check: http://localhost/health"
    echo "   💬 Chat API: http://localhost/chat"
fi
echo
echo "🔧 Management Commands:"
echo "   📊 View logs: docker-compose logs -f"
echo "   🔄 Restart: docker-compose restart"
echo "   ⏹️  Stop: docker-compose down"
echo "   ▶️  Start: docker-compose up -d"
echo "   📈 Monitor: docker-compose ps"
echo
echo "🧪 Test Your Chatbot:"
echo "   curl -X POST http://localhost/chat \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"message\": \"what is biochar?\", \"max_results\": 5}'"
echo
log_warning "⚠️  Remember to:"
echo "   1. Edit .env with your actual API keys if you haven't already"
echo "   2. Update firewall settings to allow HTTP (80) and HTTPS (443)"
echo "   3. Point your domain's DNS to this server's IP address"
echo
log_success "Deployment script completed! 🚀"