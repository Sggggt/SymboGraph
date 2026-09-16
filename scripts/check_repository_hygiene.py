"""Read-only checks for documentation links and repository artifact boundaries."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
FENCES = re.compile(r"^\s*(`{3,}|~{3,})")
LINKS = re.compile(r'!?\[[^\]\n]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+"[^"]*")?\s*\)')


def visible_markdown(source: str) -> str:
    lines, fence, math = [], None, False
    for line in source.splitlines():
        marker = FENCES.match(line)
        if marker:
            char = marker.group(1)[0]
            fence = None if fence == char else char if fence is None else fence
            continue
        if fence is not None:
            continue
        if line.strip() == '$$':
            math = not math
            continue
        if not math:
            lines.append(line)
    return '\n'.join(lines)


def anchors(source: str) -> set[str]:
    result = set(re.findall(r'<a\s+[^>]*id=["\']([^"\']+)', source))
    seen: dict[str, int] = {}
    for line in visible_markdown(source).splitlines():
        match = re.match(r'^#{1,6}\s+(.+?)\s*#*$', line)
        if not match:
            continue
        title = re.sub(r'<[^>]+>', '', match.group(1)).replace('`', '').lower()
        slug = re.sub(r'[^\w\-\s]', '', title).replace(' ', '-')
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        result.add(slug if count == 0 else f'{slug}-{count}')
    return result


def documentation_files(root: Path) -> list[Path]:
    paths = set(root.glob('README*.md'))
    paths.update((root / 'docs').rglob('*.md'))
    for area in ('apps', 'packages', 'infra', 'scripts'):
        # Only the small maintained README locations, never node_modules.
        base = root / area
        paths.update(base.glob('README.md'))
        paths.update(base.glob('*/README.md'))
        paths.update(base.glob('*/*/README.md'))
    generated = {'node_modules', '.pytest_cache', '.next', '.venv', '__pycache__'}
    public = []
    for path in paths:
        if not path.is_file() or generated.intersection(path.relative_to(root).parts):
            continue
        ignored = subprocess.run(
            ['git', 'check-ignore', '--quiet', '--no-index', str(path.relative_to(root))],
            cwd=root,
            check=False,
        ).returncode == 0
        if not ignored:
            public.append(path)
    return sorted(public)


def audit_links(root: Path) -> tuple[list[str], int]:
    root = root.resolve()
    errors, checked = [], 0
    cache: dict[Path, str] = {}
    for file in documentation_files(root):
        try:
            text = file.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            errors.append(f'{file.relative_to(root)}: invalid UTF-8')
            continue
        if text.startswith('\ufeff'):
            errors.append(f'{file.relative_to(root)}: UTF-8 BOM')
        cache[file.resolve()] = text
        for match in LINKS.finditer(visible_markdown(text)):
            href = match.group(1).strip('<>')
            url = urlsplit(href)
            if url.scheme or href.startswith('//'):
                continue
            checked += 1
            target = (file.parent / unquote(url.path)).resolve() if url.path else file.resolve()
            if not target.is_relative_to(root) or not target.exists():
                errors.append(f'{file.relative_to(root)}: missing local link {href}')
            elif url.fragment and target.suffix == '.md':
                body = cache.setdefault(target, target.read_text(encoding='utf-8'))
                if unquote(url.fragment) not in anchors(body):
                    errors.append(f'{file.relative_to(root)}: missing anchor {href}')
    return errors, checked


def audit_git(root: Path) -> list[str]:
    result = subprocess.run(['git', 'ls-files', '-ci', '--exclude-standard'], cwd=root,
        capture_output=True, text=True, check=True)
    return [f'tracked ignored artifact: {line}' for line in result.stdout.splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-clean-output', action='store_true', help='Require output/ to contain only .gitkeep.')
    args = parser.parse_args()
    errors, count = audit_links(ROOT)
    errors.extend(audit_git(ROOT))
    if args.check_clean_output:
        extras = [p.name for p in (ROOT / 'output').iterdir() if p.name != '.gitkeep']
        if extras:
            errors.append(f'output contains {len(extras)} temporary entries')
    print(json.dumps({'passed':not errors, 'documents':len(documentation_files(ROOT)),
        'local_links_checked':count, 'errors':errors}, ensure_ascii=False, indent=2))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
