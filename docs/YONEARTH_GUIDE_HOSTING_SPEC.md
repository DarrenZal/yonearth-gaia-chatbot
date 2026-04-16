# YonEarth Guide — Hosting Specification for yonearth.org

**Date:** 2026-04-15
**From:** Darren Zal (darren@earthdo.me)
**To:** Artem Nikulkov (yonearth.org hosting)
**Status:** Ready for handoff after Aaron sign-off on earthdo.me staging

---

## Overview

The YonEarth Guide is a static + API web application currently deployed at
**https://earthdo.me**. Darren owns and operates the backend (FastAPI on
earthdo.me:8000 behind nginx). Artem's role is to serve the Guide at
**https://yonearth.org/guide** — either by reverse-proxying to earthdo.me
(Mode A, recommended) or embedding it in an iframe (Mode B, fallback).

The backend never moves to yonearth.org. Artem hosts only the proxy/embed
layer and the static web assets.

---

## Mode A — Reverse Proxy (Recommended)

### Static assets

Serve the `web/` directory contents at `/guide/` on yonearth.org:

```nginx
location /guide/ {
    alias /var/www/yonearth-guide/;
    index index.html;
    try_files $uri $uri/ /guide/index.html;
}

location /guide/graph/ {
    alias /var/www/yonearth-guide/graph/;
    try_files $uri $uri/ /guide/graph/index.html;
    index index.html;
}

location /guide/podcast/ {
    alias /var/www/yonearth-guide/podcast/;
    try_files $uri $uri/ /guide/podcast/index.html;
    index index.html;
}

# Dev panel: intentionally NOT exposed under /guide/
location /guide/dev/ {
    return 404;
}
```

### API proxy

Proxy `/guide/api/*` requests to earthdo.me's backend, stripping the
`/guide` prefix:

```nginx
location /guide/api/ {
    # Strip /guide prefix — earthdo.me expects /api/*
    rewrite ^/guide/api/(.*) /api/$1 break;
    proxy_pass https://earthdo.me;
    proxy_set_header Host earthdo.me;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_http_version 1.1;
    proxy_connect_timeout 60s;
    proxy_send_timeout 60s;
    proxy_read_timeout 60s;

    # STT-specific settings (required for voice input uploads)
    client_max_body_size 10M;
    proxy_request_buffering off;
}
```

### Rate-limit trust chain (required for STT)

For per-user rate limiting to work correctly through the proxy, earthdo.me
needs to see the real client IP, not yonearth.org's IP:

**On earthdo.me (Darren configures):**
```nginx
# In http{} block — trust yonearth.org's egress IP
set_real_ip_from <YONEARTH_SERVER_IP>/32;
real_ip_header X-Forwarded-For;
limit_req_status 429;
```

**Artem provides:** the stable egress IP of yonearth.org's server. This is
the IP that earthdo.me will see on incoming proxy requests. Must be a
single `/32` — never a broad range.

**G5 gate (hard requirement before STT is enabled on /guide):** after
configuration, Artem runs the validation script from inside yonearth.org:

```bash
bash tests/smoke/test_rate_limit_isolation.sh https://earthdo.me
```

This verifies that two distinct client IPs (via X-Forwarded-For) each get
their own rate-limit bucket. STT remains disabled at `/guide/api/stt`
(returns 404) until G5 passes.

### STT disabled until G5 passes

Until the trust-chain validation passes, the frontend at `/guide/` will
NOT show the mic button (the `GET /api/stt/status` probe returns
`{"enabled": true}` from earthdo.me, but Artem should configure a
`location = /guide/api/stt` that returns 404 until G5 passes):

```nginx
# TEMPORARY — remove after G5 validation passes
location = /guide/api/stt {
    return 404;
}
```

This ensures the frontend's status probe gets 404 from yonearth.org and
hides the mic button cleanly — no error noise, no failed POST attempts.

---

## Mode B — iframe Embed (Fallback)

If Mode A's proxy setup isn't feasible, embed the Guide in an iframe:

```html
<iframe
    src="https://earthdo.me/"
    width="100%"
    height="800"
    style="border: none; min-height: 90vh;"
    allow="microphone; autoplay; clipboard-write"
    title="YonEarth Guide"
></iframe>
```

**Requirements:**
- `allow="microphone"` — needed for voice input (STT)
- `allow="autoplay"` — needed for voice output (TTS auto-play)
- `allow="clipboard-write"` — optional, for copy-to-clipboard features

**On earthdo.me (Darren configures):**
```nginx
# Allow yonearth.org to iframe the Guide
add_header Content-Security-Policy "frame-ancestors 'self' https://yonearth.org" always;
```

The `X-Frame-Options: SAMEORIGIN` header currently set on earthdo.me must
be relaxed to allow cross-origin framing from yonearth.org.

**Limitations of Mode B:**
- URL bar shows yonearth.org but the iframe content is earthdo.me
- Browser permissions (mic, autoplay) may require explicit user grant
- No deep-linking into /graph/ or /podcast/ from yonearth.org's nav

---

## Operational Boundaries

| Responsibility | Owner |
|---------------|-------|
| Backend (FastAPI, Redis, Pinecone) | Darren (earthdo.me) |
| SSL cert on earthdo.me | Darren |
| nginx on earthdo.me | Darren |
| Static assets at yonearth.org/guide/ | Artem |
| nginx proxy config on yonearth.org | Artem |
| SSL cert on yonearth.org | Artem |
| `set_real_ip_from` IP coordination | Artem provides IP, Darren configures |
| G5 trust-chain validation | Artem runs script, Darren reviews result |

---

## Cost Estimate (for Aaron)

Monthly operating costs at ~10,000 user interactions/month:

| Service | Cost | Notes |
|---------|------|-------|
| ElevenLabs TTS (Creator plan) | ~$25–30/mo | ~100k chars/mo included at $22/mo; overage ~$0.30/1k chars |
| OpenAI Whisper STT | ~$30/mo | $0.006/min; 10k × 30-sec avg = 5,000 min |
| OpenAI Chat (GPT-3.5-turbo) | ~$5/mo | Already in use; embedding costs negligible |
| **Total** | **~$60/mo** | Scales linearly with usage |

Aaron's voice is already cloned (voice ID `YcVr5DmTjJ2cEVwNiuhU`) — no
additional one-time cost. The daily STT budget cap (`STT_DAILY_BUDGET`,
default $5/day) prevents runaway costs from abuse.

---

## Deployment Checklist (Artem)

1. Receive `web/` bundle from Darren (rsync or tar)
2. Place contents at `/var/www/yonearth-guide/`
3. Add nginx location blocks per Mode A above
4. Run `nginx -t && nginx -s reload`
5. Verify: `curl https://yonearth.org/guide/` returns the landing page
6. Verify: `curl https://yonearth.org/guide/api/taxonomy` returns JSON
7. Provide egress IP to Darren for `set_real_ip_from` configuration
8. Run G5 validation script when Darren confirms IP is configured
9. Remove temporary `/guide/api/stt` 404 block after G5 passes
