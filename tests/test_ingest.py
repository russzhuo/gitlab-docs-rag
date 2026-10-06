from pathlib import Path

from ingest import load_corpus, parse_doc


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_doc_frontmatter(tmp_path):
    p = _write(tmp_path / "hr" / "pto.md",
               "---\ntitle: Time Off\nsection: hr\nsource: https://example.com/pto\n---\n\n"
               + "Flexible PTO does not require manager approval for normal vacation.\n" * 6)
    doc = parse_doc(p, tmp_path)
    assert doc is not None
    assert doc.metadata["title"] == "Time Off"
    assert doc.metadata["section"] == "hr"
    assert doc.metadata["source"] == "https://example.com/pto"
    assert doc.metadata["path"] == "hr/pto.md"


def test_parse_doc_source_comment(tmp_path):
    p = _write(tmp_path / "security" / "report.md",
               "<!-- source: https://example.com/sec | section: security -->\n\n"
               + "To report an incident, contact the security team immediately.\n" * 6)
    doc = parse_doc(p, tmp_path)
    assert doc is not None
    assert doc.metadata["source"] == "https://example.com/sec"
    assert doc.metadata["section"] == "security"
    assert "source:" not in doc.page_content


def test_parse_doc_skips_short(tmp_path):
    p = _write(tmp_path / "hr" / "short.md", "too short")
    assert parse_doc(p, tmp_path) is None


def test_load_corpus_skips_readme(tmp_path):
    _write(tmp_path / "hr" / "policy.md", "hello world " * 20)
    _write(tmp_path / "README.md", "skip me please " * 20)
    docs = load_corpus(tmp_path)
    assert [d.metadata["path"] for d in docs] == ["hr/policy.md"]
