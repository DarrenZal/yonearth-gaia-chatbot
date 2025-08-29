# Voice Integration Documentation

This document provides comprehensive information about the voice integration feature in the YonEarth Gaia Chatbot, which uses ElevenLabs text-to-speech technology to bring Gaia's responses to life.

## Overview

The voice integration allows users to hear Gaia's responses spoken aloud using a custom AI voice. This creates a more immersive and accessible experience, making the Earth's wisdom available through natural speech.

## Features

- **🎤 Custom Voice**: Uses a specially cloned voice through ElevenLabs for authentic, natural speech
- **🔊 Toggle Control**: Simple speaker button to enable/disable voice output
- **▶️ Auto-playback**: Responses automatically play when voice is enabled
- **🔄 Manual Replay**: Audio control button allows replaying responses
- **💾 Persistent Settings**: Voice preferences saved in browser localStorage
- **📝 Smart Text Processing**: Removes markdown, citations, and formatting for natural speech

## Technical Architecture

### Backend Components

#### Voice Client (`src/voice/elevenlabs_client.py`)
The core voice generation system that:
- Initializes ElevenLabs API client with custom voice ID
- Converts text responses to speech using the `eleven_multilingual_v2` model
- Preprocesses text to remove markdown, citations, and URLs
- Returns base64-encoded audio data for web delivery
- Handles errors gracefully with fallback to text-only responses

#### API Integration
Voice generation is integrated into existing chat endpoints:
- `/api/chat` and `/api/bm25/chat` accept `enable_voice` parameter
- Response includes `audio_data` field with base64-encoded MP3 audio
- Voice generation happens server-side after response text is generated
- No additional API calls needed from frontend

#### Production Server (`simple_server.py`)
The production server includes:
- Voice client initialization on startup
- Voice generation in chat handler
- Test endpoint at `/api/voice/test` for diagnostics
- Environment variable loading for API credentials

### Frontend Components

#### JavaScript Integration (`web/chat.js`)
Voice features in the chat interface:
```javascript
// Voice toggle state management
this.voiceEnabled = localStorage.getItem('voiceEnabled') === 'true';

// Include voice flag in API requests
const requestData = {
    message: userMessage,
    enable_voice: this.voiceEnabled,
    // ... other parameters
};

// Handle audio playback
if (data.audio_data && this.voiceEnabled) {
    this.playAudio(data.audio_data);
}
```

#### UI Components (`web/index.html`)
Voice controls in the interface:
- Speaker button in control panel for enabling/disabling voice
- Audio replay button appears when voice response is available
- Visual feedback showing voice status

## Configuration

### Environment Variables
```bash
# Required for voice features
ELEVENLABS_API_KEY=your-api-key-here
ELEVENLABS_VOICE_ID=your-voice-id-here
ELEVENLABS_MODEL_ID=eleven_multilingual_v2  # Optional, defaults to this
```

### API Quota Management
ElevenLabs API has character-based quotas:
- Each request consumes credits based on text length
- Monitor usage through ElevenLabs dashboard
- Text preprocessing helps reduce character count
- Graceful fallback when quota exceeded

## Usage

### Enabling Voice
1. Click the speaker button (🔊) in the chat interface
2. The button turns green when voice is enabled
3. Submit a message - the response will include audio
4. Audio plays automatically when received

### Replaying Audio
1. After a voice response, an audio button appears
2. Click the audio button (🔊) to replay the response
3. Each message's audio can be replayed independently

### Voice Settings
Voice preferences are automatically saved:
- Toggle state persists across sessions
- No login or account required
- Settings stored in browser localStorage
- Clear browser data to reset preferences

## API Usage Examples

### Chat with Voice Enabled
```bash
curl -X POST http://your-server/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about composting",
    "enable_voice": true,
    "personality": "warm_mother",
    "max_citations": 3
  }'
```

Response includes audio data:
```json
{
  "response": "Ah, composting, dear one! It's one of the most beautiful ways...",
  "audio_data": "base64-encoded-mp3-data...",
  "sources": [...],
  "success": true
}
```

### Test Voice Configuration
```bash
curl http://your-server/api/voice/test
```

Returns diagnostic information:
```json
{
  "voice_client_exists": true,
  "elevenlabs_key_set": true,
  "voice_id": "YcVr5DmTjJ2cEVwNiuhU",
  "test_generation": "Success",
  "audio_length": 22908
}
```

## Text Processing

The voice client preprocesses text for optimal speech generation:

1. **Markdown Removal**: Strips bold, italic, headers, and code formatting
2. **Citation Cleanup**: Removes [1], [2] style citations
3. **URL Removal**: Strips HTTP/HTTPS links
4. **Whitespace Normalization**: Cleans up extra spaces and line breaks
5. **Speech Optimization**: Adds pauses and ensures proper sentence endings

Example transformation:
```
Input: "**Biochar** is amazing! [1] Learn more at https://example.com"
Output: "Biochar is amazing! Learn more at"
```

## Troubleshooting

### No Audio Playing
1. Check browser console for errors
2. Ensure voice is enabled (speaker button green)
3. Verify browser supports audio playback
4. Check network tab for audio data in response

### Voice Client Not Initialized
1. Check server logs for initialization errors
2. Verify environment variables are set
3. Test with `/api/voice/test` endpoint
4. Restart service after configuration changes

### Quota Exceeded Errors
1. Check ElevenLabs dashboard for usage
2. Consider upgrading plan for more credits
3. Text preprocessing reduces but doesn't eliminate usage
4. Implement rate limiting if needed

### Audio Quality Issues
1. Ensure stable internet connection
2. Check browser audio settings
3. Try different browser if issues persist
4. Voice model uses high-quality MP3 encoding

## Browser Compatibility

Voice features work on all modern browsers:
- ✅ Chrome/Chromium (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Edge
- ⚠️ Older browsers may have limited support

## Security Considerations

- API keys are server-side only, never exposed to frontend
- Audio data transmitted as base64 to prevent XSS
- CORS headers configured for security
- Rate limiting prevents abuse

## Performance Optimization

- Voice generation happens asynchronously
- Audio preloading for smooth playback
- Caching considerations for repeated queries
- Text preprocessing reduces API usage

## Future Enhancements

Potential improvements for voice integration:
- Multiple voice options for different personalities
- Adjustable speech speed and pitch
- Streaming audio for long responses
- Offline voice synthesis option
- Voice input (speech-to-text) for queries

## Integration with Other Features

Voice works seamlessly with:
- All personality variants (warm_mother, wise_guide, earth_activist)
- Both search methods (Original and BM25)
- Custom system prompts
- All response types (episodes and books)
- Feedback system

## Monitoring and Analytics

Track voice feature usage:
- Server logs show voice generation attempts
- API test endpoint for health checks
- ElevenLabs dashboard for usage metrics
- Frontend localStorage for user preferences

## Contributing

To contribute to voice features:
1. Test changes with `/api/voice/test` endpoint
2. Ensure text preprocessing maintains quality
3. Handle errors gracefully
4. Update documentation for new features
5. Consider API quota impact

## Support

For voice-related issues:
- Check server logs: `sudo journalctl -u yonearth-gaia -f`
- Test endpoint: `curl http://your-server/api/voice/test`
- Review ElevenLabs documentation
- Submit issues to GitHub repository