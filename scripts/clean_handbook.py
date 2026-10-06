#!/usr/bin/env python3
"""Clean GitLab Handbook markdown for a RAG corpus."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

MIN_TEXT_CHARS = 500
REPO = "https://gitlab.com/gitlab-com/content-sites/handbook.git"

FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
SHORTCODE = re.compile(r"\{\{[%<]\s*/?\s*[\w.-]+(?:\s+[^}%]*)?\s*[%}]>\}\}", re.DOTALL)
ICON_TAG = re.compile(r"<i\b[^>]*>\s*</i>", re.IGNORECASE)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
HUGO_ATTR = re.compile(r"\{[.#][^}\n]*\}")
MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
VERSIONED = re.compile(r"(-v\d+|_index-old|-old)(?=\.md$)", re.IGNORECASE)
INTERNAL_HINT = re.compile(
    r"internal\.gitlab\.com|internal handbook|we are now part of|please visit our updated pages",
    re.IGNORECASE,
)


def strip_markup(text: str) -> str:
    text = FRONT_MATTER.sub("", text)
    text = HTML_COMMENT.sub("", text)
    text = SHORTCODE.sub("", text)
    text = ICON_TAG.sub("", text)
    text = MD_IMAGE.sub("", text)
    text = HUGO_ATTR.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def prose_chars(text: str) -> int:
    plain = MD_LINK.sub(r"\1", text)
    plain = re.sub(r"[#>*_\-\[\]()`]", " ", plain)
    return len(re.sub(r"\s+", "", plain))


def skip_reason(path: Path, text: str) -> str | None:
    parts = {p.lower() for p in path.parts}
    name = path.name.lower()
    if "readmes" in parts or name in {"readme.md", "reademe.md"}:
        return "personal readme"
    if "job-description-library" in parts:
        return "job description"
    if VERSIONED.search(path.name):
        return "old version"
    if INTERNAL_HINT.search(text) and prose_chars(text) < 1500:
        return "internal or redirect stub"
    if prose_chars(text) < MIN_TEXT_CHARS:
        return "too short"
    return None


def handbook_url(rel: Path) -> str:
    """Map a handbook-relative path to its public handbook.gitlab.com URL."""
    p = rel.with_suffix("")  # drop .md
    if p.name == "index":
        p = p.parent
    return f"https://handbook.gitlab.com/handbook/{p.as_posix()}/"


def iter_markdown(root: Path):
    handbook = root / "content" / "handbook"
    base = handbook if handbook.is_dir() else root
    for path in sorted(base.rglob("*.md")):
        yield base, path


def clone(dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(dest)],
        check=True,
    )
    subprocess.run(["git", "sparse-checkout", "set", "content/handbook"], cwd=dest, check=True)
    return dest


def clean(input_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    kept, dropped = [], []
    for base, path in iter_markdown(input_dir):
        text = strip_markup(path.read_text(encoding="utf-8", errors="replace"))
        reason = skip_reason(path, text)
        rel = path.relative_to(base)
        if reason:
            dropped.append(f"{reason}\t{rel.as_posix()}")
            continue
        section = rel.parts[0] if rel.parts else "handbook"
        header = f"<!-- source: {handbook_url(rel)} | section: {section} -->\n"
        dest = output_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(header + text, encoding="utf-8")
        kept.append(rel.as_posix())
    (output_dir / "NOTICE.md").write_text(
        "# Corpus provenance\n\n"
        "Content is derived from the [GitLab Handbook](https://handbook.gitlab.com)\n"
        "(https://gitlab.com/gitlab-com/content-sites/handbook), licensed under\n"
        "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).\n\n"
        "The original markdown has been modified (front-matter, shortcodes, images and\n"
        "short or duplicate pages removed) for a personal RAG demo. Any reuse must\n"
        "attribute GitLab and be shared under CC BY-SA 4.0.\n",
        encoding="utf-8",
    )
    (output_dir / "_kept.txt").write_text("\n".join(kept) + "\n", encoding="utf-8")
    (output_dir / "_dropped.txt").write_text("\n".join(dropped) + "\n", encoding="utf-8")
    print(f"kept {len(kept)}  dropped {len(dropped)}  -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clone", action="store_true")
    parser.add_argument("--clone-dir", type=Path, default=Path(".handbook-clone"))
    args = parser.parse_args()
    source = clone(args.clone_dir) if args.clone else args.input
    if source is None:
        sys.exit("pass --input or --clone")
    clean(source, args.output)


if __name__ == "__main__":
    main()