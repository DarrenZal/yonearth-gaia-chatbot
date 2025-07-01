# YonEarth Gaia Chatbot - Render Deployment Guide

## 🚀 Quick Deploy to Render

### Prerequisites
1. **OpenAI API Key** - Get from [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Pinecone API Key** - Get from [Pinecone Console](https://app.pinecone.io/)
3. **GitHub Repository** - Fork or clone this repo
4. **Render Account** - Sign up at [render.com](https://render.com)

### Step 1: Create Pinecone Index

1. Log into [Pinecone Console](https://app.pinecone.io/)
2. Create new index:
   - **Name**: `yonearth-episodes`
   - **Environment**: `gcp-starter` (free tier)
   - **Dimensions**: `1536` (for OpenAI text-embedding-3-small)
   - **Metric**: `cosine`
   - **Pod Type**: `starter` (free)

### Step 2: Deploy to Render

#### Option A: One-Click Deploy with Blueprint
1. Click this button: [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/your-username/yonearth-chatbot)
2. Connect your GitHub account
3. Select the repository
4. Render will use the `render.yaml` blueprint

#### Option B: Manual Deploy
1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure:
   - **Name**: `yonearth-gaia-chat`
   - **Environment**: `Python`
   - **Build Command**: 
     ```bash
     pip install -r requirements.txt
     python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
     ```
   - **Start Command**: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/health`

### Step 3: Configure Environment Variables

Add these environment variables in Render dashboard:

```bash
# Required API Keys
OPENAI_API_KEY=sk-your-key-here
PINECONE_API_KEY=your-pinecone-key-here

# OpenAI Configuration
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Pinecone Configuration
PINECONE_ENVIRONMENT=gcp-starter
PINECONE_INDEX_NAME=yonearth-episodes

# API Settings
ALLOWED_ORIGINS=https://yonearth.org,https://your-domain.com
RATE_LIMIT_PER_MINUTE=10
LOG_LEVEL=INFO

# Gaia Personality
GAIA_PERSONALITY_VARIANT=warm_mother
GAIA_TEMPERATURE=0.7
GAIA_MAX_TOKENS=1000

# Processing Settings
EPISODES_TO_PROCESS=172
CHUNK_SIZE=500
CHUNK_OVERLAP=50
DEBUG=false

# Redis (set to placeholder for now)
REDIS_URL=redis://localhost:6379
```

### Step 4: Deploy & Test

1. Click **"Create Web Service"**
2. Wait for deployment (5-10 minutes first time)
3. Your API will be available at: `https://your-app-name.onrender.com`

### Step 5: Test Deployment

```bash
# Test health endpoint
curl https://your-app-name.onrender.com/health

# Test chat endpoint
curl -X POST https://your-app-name.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "what is biochar?",
    "max_results": 5,
    "session_id": "test-session"
  }'
```

## 🔧 Configuration Options

### Episode Processing
- **EPISODES_TO_PROCESS**: `172` (all episodes) or `20` (MVP subset)
- **CHUNK_SIZE**: `500` tokens per chunk
- **CHUNK_OVERLAP**: `50` tokens overlap between chunks

### Performance Tuning
- **Rate Limiting**: `RATE_LIMIT_PER_MINUTE=10`
- **Gaia Temperature**: `0.7` (creativity vs accuracy)
- **Max Tokens**: `1000` (response length)

### Adding Redis Caching (Optional +$7/month)
Uncomment the Redis service in `render.yaml`:
```yaml
- type: redis
  name: yonearth-redis
  plan: starter
  maxmemoryPolicy: allkeys-lru
```

## 🌐 Frontend Integration

### Simple HTML/JavaScript
```html
<!DOCTYPE html>
<html>
<head>
    <title>Chat with Gaia</title>
</head>
<body>
    <div id="chat-container">
        <div id="messages"></div>
        <input type="text" id="user-input" placeholder="Ask Gaia a question...">
        <button onclick="sendMessage()">Send</button>
    </div>

    <script>
        const API_URL = 'https://your-app-name.onrender.com';
        
        async function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            if (!message) return;
            
            // Show user message
            addMessage('user', message);
            input.value = '';
            
            try {
                const response = await fetch(`${API_URL}/chat`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        max_results: 5,
                        session_id: 'web-session'
                    })
                });
                
                const data = await response.json();
                
                // Show Gaia's response
                addMessage('gaia', data.response);
                
                // Show citations
                if (data.citations && data.citations.length > 0) {
                    showCitations(data.citations);
                }
                
            } catch (error) {
                addMessage('error', 'Sorry, I encountered an error. Please try again.');
                console.error('Error:', error);
            }
        }
        
        function addMessage(sender, text) {
            const messages = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            messageDiv.textContent = text;
            messages.appendChild(messageDiv);
            messages.scrollTop = messages.scrollHeight;
        }
        
        function showCitations(citations) {
            const citationsDiv = document.createElement('div');
            citationsDiv.className = 'citations';
            citationsDiv.innerHTML = '<h4>Referenced Episodes:</h4>';
            
            citations.forEach(citation => {
                const citationP = document.createElement('p');
                citationP.innerHTML = `
                    <a href="${citation.url}" target="_blank">
                        Episode ${citation.episode_number}: ${citation.title}
                    </a>
                `;
                citationsDiv.appendChild(citationP);
            });
            
            document.getElementById('messages').appendChild(citationsDiv);
        }
        
        // Allow Enter key to send message
        document.getElementById('user-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
    
    <style>
        #chat-container {
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 10px;
        }
        
        #messages {
            height: 400px;
            overflow-y: auto;
            border: 1px solid #eee;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
        }
        
        .message {
            margin-bottom: 10px;
            padding: 8px 12px;
            border-radius: 5px;
        }
        
        .message.user {
            background-color: #e3f2fd;
            text-align: right;
        }
        
        .message.gaia {
            background-color: #f1f8e9;
        }
        
        .message.error {
            background-color: #ffebee;
            color: #c62828;
        }
        
        .citations {
            background-color: #fafafa;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 0.9em;
        }
        
        .citations a {
            color: #2e7d32;
            text-decoration: none;
        }
        
        .citations a:hover {
            text-decoration: underline;
        }
        
        #user-input {
            width: 70%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        
        button {
            width: 25%;
            padding: 10px;
            background-color: #4caf50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        
        button:hover {
            background-color: #45a049;
        }
    </style>
</body>
</html>
```

## 📊 Monitoring & Logs

### View Logs
1. Go to Render Dashboard → Your Service
2. Click **"Logs"** tab
3. Monitor initialization and request processing

### Common Issues
- **503 Service Unavailable**: API keys missing or invalid
- **Slow first response**: Pinecone index building (normal)
- **Memory errors**: Try reducing `EPISODES_TO_PROCESS` to `50`

### Health Check
Monitor: `https://your-app-name.onrender.com/health`

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "rag_initialized": true,
  "vectorstore_stats": {"total_vector_count": 1500},
  "gaia_personality": "warm_mother"
}
```

## 💰 Cost Estimate

### Render Costs
- **Web Service**: $7/month (Starter plan)
- **Redis** (optional): $7/month
- **Total**: $7-14/month

### API Costs (estimated)
- **OpenAI GPT-3.5-turbo**: ~$0.002 per chat
- **OpenAI Embeddings**: ~$0.01 for initial setup, minimal ongoing
- **Pinecone**: Free tier (up to 1M vectors)

**Total estimated cost**: $7-15/month for unlimited public access

## 🔒 Security Notes

- API keys are securely stored in Render environment variables
- CORS configured for specific origins
- Rate limiting prevents abuse
- No user data stored permanently
- HTTPS enforced automatically

## 🚀 Next Steps

1. **Custom Domain**: Configure your domain in Render settings
2. **WordPress Plugin**: Use the API endpoints for WordPress integration
3. **Analytics**: Add monitoring with tools like LogRocket or Sentry
4. **Caching**: Enable Redis for better performance
5. **Scaling**: Upgrade to higher Render plans as usage grows

Your hybrid search RAG chatbot will be live and accessible to anyone worldwide!