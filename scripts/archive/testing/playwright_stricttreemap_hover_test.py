import asyncio
from pathlib import Path

from playwright.async_api import async_playwright


URL = "https://gaiaai.xyz/YonEarth/graph/GraphRAG3D_EmbeddingView_stricttreemap.html"
SCREENSHOT_DIR = Path("playwright_artifacts")
SCREENSHOT_DIR.mkdir(exist_ok=True)


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1500, "height": 900})

        await page.goto(URL, wait_until="networkidle", timeout=180_000)
        await page.wait_for_timeout(4000)

        # Ensure the 2D SVG is active and polygons are present
        await page.wait_for_selector("#svg-container-2d.active", timeout=20_000)
        await page.wait_for_selector("path.l3-polygon", timeout=20_000)

        # Prefer a specific cluster if present, otherwise pick the first
        target_label = "Sound and Spirituality"
        label_locator = page.locator(f'text.l3-label', has_text=target_label)
        polygon_locator = None

        if await label_locator.count():
            cluster_id = await label_locator.first.get_attribute("data-id")
            polygon_locator = page.locator(f'path.l3-polygon[data-id=\"{cluster_id}\"]').first
        else:
            polygon_locator = page.locator("path.l3-polygon").first
            cluster_id = await polygon_locator.get_attribute("data-id")
            label_locator = page.locator(f'text.l3-label[data-id=\"{cluster_id}\"]')

        # Capture pre-hover state
        before_label_opacity = await label_locator.evaluate("el => getComputedStyle(el).opacity")
        before_label_display = await label_locator.evaluate("el => getComputedStyle(el).display")
        before_label_style = await label_locator.evaluate("el => el.getAttribute('style')")
        before_poly_opacity = await polygon_locator.evaluate("el => getComputedStyle(el).fillOpacity")
        before_poly_attr = await polygon_locator.evaluate("el => el.getAttribute('fill-opacity')")
        await page.screenshot(path=SCREENSHOT_DIR / "before_hover.png")

        # Hover the polygon (force avoids overlay interference)
        await polygon_locator.hover(force=True)
        await page.wait_for_timeout(800)

        after_label_opacity = await label_locator.evaluate("el => getComputedStyle(el).opacity")
        after_label_display = await label_locator.evaluate("el => getComputedStyle(el).display")
        after_label_style = await label_locator.evaluate("el => el.getAttribute('style')")
        after_poly_opacity = await polygon_locator.evaluate("el => getComputedStyle(el).fillOpacity")
        after_poly_attr = await polygon_locator.evaluate("el => el.getAttribute('fill-opacity')")
        await page.screenshot(path=SCREENSHOT_DIR / "after_hover.png")

        print(f"Cluster ID: {cluster_id}")
        print(f"Polygon fill opacity: before {before_poly_opacity} (attr {before_poly_attr}), after {after_poly_opacity} (attr {after_poly_attr})")
        print(f"Label -> before opacity {before_label_opacity}, display {before_label_display}, style attr: {before_label_style}")
        print(f"Label -> after  opacity {after_label_opacity}, display {after_label_display}, style attr: {after_label_style}")

        # Simple assertion: label should be hidden (opacity 0 or display none)
        hidden = (after_label_opacity == "0") or (after_label_display == "none")
        print(f"Hidden after hover: {hidden}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
