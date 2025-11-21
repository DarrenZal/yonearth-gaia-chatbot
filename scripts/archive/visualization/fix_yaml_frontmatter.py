#!/usr/bin/env python3
"""
Fix YAML frontmatter in markdown files - quote values containing colons
"""
import re
from pathlib import Path

def fix_yaml_field(content, field_name):
    """Quote field values containing colons if not already quoted"""
    # Match: field: value with colon (not already quoted)
    pattern = rf'^{field_name}: ([^"\n].+:.*?)$'

    def replace_match(match):
        value = match.group(1).strip()
        # Only quote if not already quoted and contains colon
        if ':' in value and not (value.startswith('"') and value.endswith('"')):
            return f'{field_name}: "{value}"'
        return match.group(0)

    return re.sub(pattern, replace_match, content, flags=re.MULTILINE)

def fix_markdown_file(file_path):
    """Fix YAML frontmatter in a single markdown file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Fix common fields that might have colons
        original = content
        content = fix_yaml_field(content, 'title')
        content = fix_yaml_field(content, 'name')
        content = fix_yaml_field(content, 'guest')

        # Only write if changed
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    content_dir = Path("/home/claudeuser/yonearth-gaia-chatbot/wiki-quartz/content")

    fixed_count = 0
    for md_file in content_dir.rglob("*.md"):
        if fix_markdown_file(md_file):
            fixed_count += 1

    print(f"Fixed {fixed_count} markdown files")

if __name__ == "__main__":
    main()
