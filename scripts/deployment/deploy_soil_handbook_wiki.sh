#!/bin/bash

# Deploy Soil Stewardship Handbook Wiki to http://152.53.194.214/soil-handbook/

set -e

echo "=========================================="
echo "Deploying Soil Stewardship Handbook Wiki"
echo "=========================================="

# Check if wiki exists
if [ ! -d "/home/claudeuser/yonearth-gaia-chatbot/web/soil-handbook-wiki" ]; then
    echo "Error: Wiki directory not found!"
    exit 1
fi

# Install wiki-to-html converter if needed (using markdown-to-html or similar)
# For now, we'll use a simple Python-based approach with markdown library

echo ""
echo "Step 1: Converting Markdown to HTML..."

# Create output directory
HTML_DIR="/var/www/html/soil-handbook"
sudo mkdir -p "$HTML_DIR"

# Install markdown converter if not present
if ! python3 -c "import markdown" 2>/dev/null; then
    echo "Installing markdown library..."
    sudo pip3 install markdown pymdown-extensions
fi

# Create conversion script
cat > /tmp/convert_wiki_to_html.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
import os
import sys
import markdown
from pathlib import Path
import re

def convert_wikilinks_to_html(content):
    """Convert [[WikiLinks]] to HTML links."""
    # Pattern: [[Page Name]] or [[Page Name|Display Text]]
    pattern = r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]'

    def replace_link(match):
        page = match.group(1)
        display = match.group(2) if match.group(2) else page
        # Convert to URL-friendly format
        url = page.replace(' ', '_') + '.html'
        return f'<a href="{url}">{display}</a>'

    return re.sub(pattern, replace_link, content)

def add_html_wrapper(title, body_html):
    """Add HTML structure around the content."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Soil Stewardship Handbook Wiki</title>
    <style>
        :root {{
            --bg-color: #1a1a1a;
            --text-color: #e0e0e0;
            --link-color: #4a9eff;
            --heading-color: #7cb342;
            --border-color: #333;
            --code-bg: #2a2a2a;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: var(--bg-color);
            color: var(--text-color);
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: var(--heading-color);
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }}
        h1 {{
            font-size: 2.5em;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 0.3em;
        }}
        h2 {{
            font-size: 2em;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.2em;
        }}
        a {{
            color: var(--link-color);
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        ul, ol {{
            padding-left: 2em;
        }}
        code {{
            background: var(--code-bg);
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background: var(--code-bg);
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        hr {{
            border: none;
            border-top: 1px solid var(--border-color);
            margin: 2em 0;
        }}
        .metadata {{
            background: var(--code-bg);
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 0.9em;
        }}
        .nav {{
            background: var(--code-bg);
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .nav a {{
            margin-right: 20px;
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
            text-align: center;
            font-size: 0.9em;
            color: #888;
        }}
    </style>
</head>
<body>
    <div class="nav">
        <a href="Index.html">🏠 Home</a>
        <a href="People_Index.html">👥 People</a>
        <a href="Concepts_Index.html">💡 Concepts</a>
        <a href="Organizations_Index.html">🏢 Organizations</a>
        <a href="Practices_Index.html">🌱 Practices</a>
    </div>
    {body_html}
    <div class="footer">
        <p>Soil Stewardship Handbook Knowledge Graph | Generated from structured extraction</p>
        <p><a href="/">← Back to YonEarth Chatbot</a></p>
    </div>
</body>
</html>"""

def extract_frontmatter(content):
    """Extract YAML frontmatter and content."""
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            body = parts[2].strip()
            return frontmatter, body
    return '', content

def frontmatter_to_html(frontmatter):
    """Convert frontmatter to HTML metadata display."""
    if not frontmatter:
        return ''

    lines = frontmatter.split('\n')
    html = '<div class="metadata">\n'
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            html += f'<strong>{key.strip()}:</strong> {value.strip()}<br>\n'
    html += '</div>\n'
    return html

def convert_file(input_file, output_file):
    """Convert a single markdown file to HTML."""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract frontmatter
    frontmatter, body = extract_frontmatter(content)

    # Convert wikilinks
    body = convert_wikilinks_to_html(body)

    # Convert markdown to HTML
    md = markdown.Markdown(extensions=['extra', 'tables', 'toc'])
    body_html = md.convert(body)

    # Add frontmatter as metadata
    if frontmatter:
        body_html = frontmatter_to_html(frontmatter) + body_html

    # Get title from first heading or filename
    title_match = re.search(r'^#\s+(.+)$', body, re.MULTILINE)
    title = title_match.group(1) if title_match else input_file.stem

    # Wrap in HTML structure
    full_html = add_html_wrapper(title, body_html)

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_html)

def main():
    wiki_dir = Path('/home/claudeuser/yonearth-gaia-chatbot/web/soil-handbook-wiki')
    output_dir = Path('/var/www/html/soil-handbook')

    # Convert all markdown files
    for md_file in wiki_dir.rglob('*.md'):
        # Preserve directory structure
        rel_path = md_file.relative_to(wiki_dir)
        output_file = output_dir / rel_path.with_suffix('.html')

        print(f"Converting: {rel_path}")
        convert_file(md_file, output_file)

    print(f"\nConversion complete! {len(list(wiki_dir.rglob('*.md')))} files converted.")

if __name__ == '__main__':
    main()
PYTHON_SCRIPT

# Run conversion
sudo python3 /tmp/convert_wiki_to_html.py

echo ""
echo "Step 2: Configuring nginx..."

# Create nginx configuration
sudo tee /etc/nginx/sites-available/soil-handbook << 'NGINX_CONFIG'
server {
    listen 80;
    server_name 152.53.194.214;

    # Soil Handbook Wiki
    location /soil-handbook/ {
        alias /var/www/html/soil-handbook/;
        index Index.html index.html;
        try_files $uri $uri/ =404;

        # Enable directory listing for debugging (optional)
        # autoindex on;
    }
}
NGINX_CONFIG

# Enable site if not already enabled
if [ ! -L "/etc/nginx/sites-enabled/soil-handbook" ]; then
    sudo ln -sf /etc/nginx/sites-available/soil-handbook /etc/nginx/sites-enabled/
fi

# Test nginx configuration
echo ""
echo "Step 3: Testing nginx configuration..."
sudo nginx -t

# Reload nginx
echo ""
echo "Step 4: Reloading nginx..."
sudo systemctl reload nginx

# Set permissions
echo ""
echo "Step 5: Setting permissions..."
sudo chown -R www-data:www-data /var/www/html/soil-handbook
sudo chmod -R 755 /var/www/html/soil-handbook

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "Your Soil Stewardship Handbook wiki is now live at:"
echo "http://152.53.194.214/soil-handbook/"
echo ""
echo "Pages generated: $(find /var/www/html/soil-handbook -name '*.html' | wc -l)"
echo ""
echo "Quick links:"
echo "  - Index: http://152.53.194.214/soil-handbook/Index.html"
echo "  - People: http://152.53.194.214/soil-handbook/_indexes/People_Index.html"
echo "  - Concepts: http://152.53.194.214/soil-handbook/_indexes/Concepts_Index.html"
echo ""
