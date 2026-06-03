const { expect, test } = require('@playwright/test');
const path = require('node:path');

const repoRoot = path.resolve(__dirname, '../..');
const d3BundlePath = path.join(repoRoot, 'node_modules/d3/dist/d3.min.js');

test.beforeEach(async ({ page }) => {
  const consoleErrors = [];
  const failedRequests = [];

  await page.route('https://d3js.org/d3.v7.min.js', async route => {
    await route.fulfill({
      path: d3BundlePath,
      contentType: 'application/javascript; charset=utf-8',
    });
  });

  page.on('console', message => {
    if (message.type() === 'error') {
      consoleErrors.push(message.text());
    }
  });
  page.on('pageerror', error => {
    consoleErrors.push(error.message);
  });
  page.on('requestfailed', request => {
    failedRequests.push(`${request.method()} ${request.url()} ${request.failure()?.errorText || ''}`.trim());
  });

  page.checkNoGuideRuntimeErrors = async () => {
    await page.waitForTimeout(250);
    expect(failedRequests, 'no failed network requests').toEqual([]);
    expect(consoleErrors, 'no browser console/page errors').toEqual([]);
  };
});

async function graphFrame(page) {
  const iframe = page.locator('#kgIframe');
  await expect(iframe).toBeVisible();
  const frame = await iframe.elementHandle().then(handle => handle.contentFrame());
  expect(frame, 'knowledge graph iframe should resolve').toBeTruthy();
  return frame;
}

async function waitForGraph(frame) {
  await frame.locator('g.node').first().waitFor({ state: 'visible', timeout: 30_000 });
  await expect(frame.locator('#loading-overlay')).toBeHidden();
}

test('Guide embed mode hides welcome chrome and renders the knowledge graph', async ({ page, isMobile }) => {
  await page.goto('/guide/?embed=oasis');

  await expect(page.locator('body')).toHaveClass(/embed/);
  await expect(page.locator('body')).toHaveClass(/embed-oasis/);
  await expect(page.locator('#welcomeBanner')).toBeHidden();
  await expect(page.locator('#mainContainer')).toBeVisible();

  if (isMobile) {
    await page.locator('.mobile-tab[data-tab="explore"]').click();
  }

  const frame = await graphFrame(page);
  await waitForGraph(frame);

  await expect(frame.locator('.strip-label', { hasText: 'Categories' })).toBeVisible();
  await expect(frame.locator('.strip-label', { hasText: 'Topics' })).toBeVisible();
  await expect(frame.locator('.strip-label', { hasText: 'Types' })).toHaveCount(1);

  const renderedNodes = await frame.locator('g.node').count();
  expect(renderedNodes).toBeGreaterThan(20);

  await page.checkNoGuideRuntimeErrors();
});

test('knowledge graph data keeps Wele Waters canonical and removes Waylay node links', async ({ request }) => {
  const response = await request.get('/api/knowledge-graph/data');
  expect(response.ok()).toBeTruthy();
  const graph = await response.json();

  const nodeNames = new Set(graph.nodes.map(node => node.name));
  const nodeIds = new Set(graph.nodes.map(node => node.id));
  const wele = graph.nodes.find(node => node.name === 'Wele Waters');

  expect(wele, 'canonical Wele Waters node').toBeTruthy();
  expect(wele.type).toBe('PRODUCT');
  expect(wele.aliases || []).toContain('Waylay Waters');
  expect(nodeNames.has('Waylay Waters'), 'stale Waylay Waters node name').toBe(false);
  expect(nodeIds.has('Waylay Waters'), 'stale Waylay Waters node id').toBe(false);

  const orphanLinks = graph.links.filter(link => {
    const source = typeof link.source === 'string' ? link.source : link.source?.id;
    const target = typeof link.target === 'string' ? link.target : link.target?.id;
    return !nodeIds.has(source) || !nodeIds.has(target);
  });
  const waylayLinks = graph.links.filter(link => {
    const source = typeof link.source === 'string' ? link.source : link.source?.id;
    const target = typeof link.target === 'string' ? link.target : link.target?.id;
    return source === 'Waylay Waters' || target === 'Waylay Waters';
  });

  expect(orphanLinks, 'no orphaned graph links').toEqual([]);
  expect(waylayLinks, 'no links reference stale Waylay Waters id').toEqual([]);
});

test('resource requests from the Guide select and visibly highlight hidden resource types', async ({ page, isMobile }) => {
  await page.goto('/guide/?embed=oasis');
  if (isMobile) {
    await page.locator('.mobile-tab[data-tab="explore"]').click();
  }
  const frame = await graphFrame(page);
  await waitForGraph(frame);

  const selected = await frame.evaluate(async () => {
    const requestId = 'e2e-wele-waters';
    const response = new Promise(resolve => {
      window.addEventListener('message', event => {
        if (
          event.data?.requestId === requestId &&
          (event.data.type === 'resourceSelected' || event.data.type === 'resourceError')
        ) {
          resolve(event.data);
        }
      });
    });

    window.postMessage({
      type: 'requestResource',
      requestId,
      query: { name: 'Wele Waters' },
    }, window.location.origin);

    const result = await response;
    await new Promise(resolve => setTimeout(resolve, 900));

    const highlighted = [...document.querySelectorAll('g.node.highlighted')].map(node => {
      const datum = node.__data__ || {};
      return datum.name || datum.id;
    });

    return { result, highlighted };
  });

  expect(selected.result.type).toBe('resourceSelected');
  expect(selected.result.resource.name).toBe('Wele Waters');
  expect(selected.highlighted).toContain('Wele Waters');

  const highlightedStroke = await frame.evaluate(() => {
    const nodeGroup = [...document.querySelectorAll('g.node.highlighted')]
      .find(node => {
        const datum = node.__data__ || {};
        return datum.name === 'Wele Waters' || datum.id === 'Wele Waters';
      });
    const circle = nodeGroup?.querySelector('circle');
    if (!circle) return null;
    const styles = getComputedStyle(circle);
    return {
      stroke: styles.stroke,
      width: styles.strokeWidth,
      filter: styles.filter,
    };
  });

  expect(highlightedStroke).toBeTruthy();
  expect(highlightedStroke.width).toBe('5px');
  expect(highlightedStroke.filter).toContain('drop-shadow');

  await page.checkNoGuideRuntimeErrors();
});

test('mobile embed keeps chat and explore controls usable without overlapping panels', async ({ page, isMobile }) => {
  test.skip(!isMobile, 'mobile layout assertion only runs in the mobile project');

  await page.goto('/guide/?embed=oasis');

  await expect(page.locator('.mobile-tabs')).toBeVisible();
  await expect(page.locator('.mobile-tab[data-tab="chat"]')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('.guide-chat-panel')).toBeVisible();

  await page.locator('.mobile-tab[data-tab="explore"]').click();
  await expect(page.locator('.mobile-tab[data-tab="explore"]')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('.guide-kg-panel')).toBeVisible();

  const frame = await graphFrame(page);
  await waitForGraph(frame);
  await expect(frame.locator('#type-filter-strip')).toBeHidden();
  await expect(frame.locator('.strip-label', { hasText: 'Categories' })).toBeVisible();
  await expect(frame.locator('.strip-label', { hasText: 'Topics' })).toBeVisible();

  const overlaps = await page.evaluate(() => {
    const visibleBoxes = [...document.querySelectorAll('.mobile-tabs, .guide-chat-panel, .guide-kg-panel')]
      .filter(el => getComputedStyle(el).display !== 'none' && getComputedStyle(el).visibility !== 'hidden')
      .map(el => {
        const rect = el.getBoundingClientRect();
        return { selector: el.className, top: rect.top, bottom: rect.bottom, left: rect.left, right: rect.right };
      });

    const badPairs = [];
    for (let i = 0; i < visibleBoxes.length; i += 1) {
      for (let j = i + 1; j < visibleBoxes.length; j += 1) {
        const a = visibleBoxes[i];
        const b = visibleBoxes[j];
        const horizontal = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left));
        const vertical = Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
        if (horizontal > 8 && vertical > 8) badPairs.push([a.selector, b.selector, horizontal, vertical]);
      }
    }
    return badPairs;
  });

  expect(overlaps, 'visible mobile panels should not materially overlap').toEqual([]);
  await page.checkNoGuideRuntimeErrors();
});
