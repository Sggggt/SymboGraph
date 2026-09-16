import importlib.util
from pathlib import Path


def checker():
    path = Path(__file__).resolve().parents[3] / 'scripts/check_repository_hygiene.py'
    spec = importlib.util.spec_from_file_location('repository_hygiene', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_links_fragments_fences_and_unicode(tmp_path):
    module = checker()
    (tmp_path / 'README.md').write_text('[文档](docs/guide.md#使用说明)\n[外部](https://example.invalid/)\n'
        '```text\n[example](absent.md)\n```\n', encoding='utf-8')
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs/guide.md').write_text('# 使用说明\n\n[主页](../README.md)\n', encoding='utf-8')
    assert module.audit_links(tmp_path) == ([], 2)


def test_missing_and_escaping_paths_are_reported(tmp_path):
    module = checker()
    (tmp_path / 'README.md').write_text('[a](missing.md) [b](../outside.md) [c](#absent)', encoding='utf-8')
    issues, count = module.audit_links(tmp_path)
    assert count == 3 and len(issues) == 3


def test_duplicate_headings_and_explicit_anchors():
    module = checker()
    assert module.anchors('# 标题\n## 标题\n<a id="explicit"></a>') == {'标题', '标题-1', 'explicit'}


def test_generated_cache_readmes_are_not_repository_documents(tmp_path):
    module = checker()
    cache = tmp_path / 'apps/api/.pytest_cache'
    cache.mkdir(parents=True)
    (cache / 'README.md').write_text('[cache-only](missing-file.md)', encoding='utf-8')
    (tmp_path / 'README.md').write_text('# Project\n', encoding='utf-8')
    assert module.documentation_files(tmp_path) == [tmp_path / 'README.md']
    assert module.audit_links(tmp_path) == ([], 0)
