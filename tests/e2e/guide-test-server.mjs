import { createReadStream, existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '../..');
const webRoot = path.join(repoRoot, 'web');
const kgDataPath = path.join(repoRoot, 'data/knowledge_graph/visualization_data.json');
const port = Number(process.env.GUIDE_TEST_PORT || 8787);

const mimeTypes = new Map([
  ['.css', 'text/css; charset=utf-8'],
  ['.html', 'text/html; charset=utf-8'],
  ['.ico', 'image/x-icon'],
  ['.js', 'application/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.svg', 'image/svg+xml'],
  ['.webp', 'image/webp'],
]);

const bm25Stub = {
  response: [
    'Composting improves soil structure and carbon cycling.',
    'Episode 120 discusses biochar and soil resilience.',
    'The Soil Stewardship Handbook gives step-by-step composting practices.',
    'Wele Waters is a Y On Earth community resource.',
  ].join(' '),
  sources: [
    {
      content_type: 'episode',
      episode_number: '120',
      title: 'Episode 120 - Rowdy Yeatts, Founder & CEO, High Plains Biochar',
      guest_name: 'Rowdy Yeatts',
      url: 'https://yonearth.org/episode-120-rowdy-yeatts/',
      content_preview: 'Biochar and soil resilience.',
    },
    {
      content_type: 'book',
      book_title: 'Soil Stewardship Handbook',
      author: 'Aaron William Perry',
      chapter_number: 10,
      chapter_title: 'Composting',
      title: 'Soil Stewardship Handbook - Chapter 10',
      url: 'https://yonearth.org/soil-stewardship-handbook/',
      ebook_url: 'https://yonearth.org/soil-stewardship-handbook/',
      audiobook_url: 'https://yonearth.org/soil-stewardship-handbook-audio/',
      print_url: 'https://yonearth.org/soil-stewardship-handbook-print/',
      content_preview: 'Composting as a soil-building practice.',
    },
  ],
  citations: [],
  episode_references: ['120', 'Book: Soil Stewardship Handbook'],
  search_method_used: 'stub',
  documents_retrieved: 2,
  bm25_stats: {},
  performance_stats: {},
  processing_time: 0,
};

const recommendationStub = {
  conversation_topics: ['soil', 'composting'],
  recommendations: [
    {
      content_type: 'episode',
      episode_number: '120',
      title: 'Episode 120 - Rowdy Yeatts, Founder & CEO, High Plains Biochar',
      guest_name: 'Rowdy Yeatts',
      url: 'https://yonearth.org/episode-120-rowdy-yeatts/',
    },
    {
      content_type: 'book',
      book_title: 'Soil Stewardship Handbook',
      author: 'Aaron William Perry',
      chapter_number: 10,
      title: 'Soil Stewardship Handbook - Chapter 10',
      url: 'https://yonearth.org/soil-stewardship-handbook/',
      ebook_url: 'https://yonearth.org/soil-stewardship-handbook/',
    },
  ],
};

function send(res, status, body, contentType = 'text/plain; charset=utf-8') {
  res.writeHead(status, {
    'content-type': contentType,
    'cache-control': 'no-store',
  });
  res.end(body);
}

function sendJson(res, status, body) {
  send(res, status, JSON.stringify(body), 'application/json; charset=utf-8');
}

function sendFile(res, filePath) {
  if (!filePath.startsWith(webRoot) && filePath !== kgDataPath) {
    send(res, 403, 'Forbidden');
    return;
  }
  if (!existsSync(filePath)) {
    send(res, 404, 'Not found');
    return;
  }

  const ext = path.extname(filePath);
  res.writeHead(200, {
    'content-type': mimeTypes.get(ext) || 'application/octet-stream',
    'cache-control': 'no-store',
  });
  createReadStream(filePath).pipe(res);
}

function guideFilePath(urlPath) {
  const rel = decodeURIComponent(urlPath.replace(/^\/guide\/?/, ''));
  const normalizedRel = rel === '' ? 'index.html' : rel;
  return path.normalize(path.join(webRoot, normalizedRel));
}

const server = createServer(async (req, res) => {
  const parsed = new URL(req.url || '/', `http://${req.headers.host || '127.0.0.1'}`);
  const urlPath = parsed.pathname;

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  if (urlPath === '/') {
    res.writeHead(302, { location: '/guide/' });
    res.end();
    return;
  }

  if (urlPath === '/api/knowledge-graph/data' || urlPath === '/data/knowledge_graph/visualization_data.json') {
    sendFile(res, kgDataPath);
    return;
  }

  if (urlPath === '/api/knowledge-graph/episodes-books') {
    sendJson(res, 200, { episodes: [], books: [] });
    return;
  }

  if (urlPath.startsWith('/api/knowledge-graph/search')) {
    sendJson(res, 200, { results: [] });
    return;
  }

  if (urlPath === '/api/bm25/health' || urlPath === '/bm25/health') {
    sendJson(res, 200, { status: 'ok', search_method: 'stub' });
    return;
  }

  if (urlPath === '/api/bm25/chat' || urlPath === '/bm25/chat') {
    if (req.method !== 'POST') {
      sendJson(res, 405, { error: 'method_not_allowed' });
      return;
    }
    req.resume();
    sendJson(res, 200, bm25Stub);
    return;
  }

  if (urlPath === '/api/conversation-recommendations') {
    if (req.method !== 'POST') {
      sendJson(res, 405, { error: 'method_not_allowed' });
      return;
    }
    req.resume();
    sendJson(res, 200, recommendationStub);
    return;
  }

  if (urlPath === '/api/feedback') {
    if (req.method !== 'POST') {
      sendJson(res, 405, { error: 'method_not_allowed' });
      return;
    }
    req.resume();
    sendJson(res, 200, { ok: true });
    return;
  }

  if (urlPath === '/api/stt/status') {
    sendJson(res, 200, { available: false, enabled: false });
    return;
  }

  if (urlPath === '/YonEarth/data/top_entities.json') {
    sendJson(res, 200, {
      'Wele Waters': { type: 'PRODUCT' },
      'Soil Werks': { type: 'PRODUCT' },
      'Soil Stewardship Handbook': { type: 'BOOK' },
    });
    return;
  }

  if (urlPath === '/guide/yoe_taxonomy.json') {
    sendFile(res, path.join(webRoot, 'data/yoe_taxonomy.json'));
    return;
  }

  if (urlPath === '/favicon.ico') {
    sendFile(res, path.join(webRoot, 'favicon.ico'));
    return;
  }

  if (urlPath.startsWith('/guide/')) {
    const filePath = guideFilePath(urlPath);
    sendFile(res, filePath);
    return;
  }

  if (urlPath === '/healthz') {
    const body = await readFile(path.join(repoRoot, 'package.json'), 'utf8');
    sendJson(res, 200, { ok: true, package: JSON.parse(body).name });
    return;
  }

  send(res, 404, 'Not found');
});

server.listen(port, '127.0.0.1', () => {
  console.error(`Guide test server listening on http://127.0.0.1:${port}/guide/`);
});

process.on('SIGTERM', () => server.close(() => process.exit(0)));
process.on('SIGINT', () => server.close(() => process.exit(0)));
