# Data

## Corpus

The corpus is generated from the public [GitLab Handbook](https://handbook.gitlab.com)
by [scripts/clean_handbook.py](../scripts/clean_handbook.py):

```bash
python scripts/clean_handbook.py --clone --output data
```

This sparse-clones the handbook, strips front-matter / shortcodes / images, drops
short or internal pages, and writes cleaned markdown into `data/` — one file per
handbook page, each prefixed with a `<!-- source: ... | department: ... -->` header
carrying the original URL and the top-level section (e.g. `people-group`, `finance`,
`engineering`).

```text
data/
  people-group/
  finance/
  engineering/
  security/
  ...
  NOTICE.md      # provenance + license
  _kept.txt      # which files were kept
  _dropped.txt   # which files were dropped and why
```

`data/` and `.handbook-clone/` are gitignored: the corpus is generated locally, not
committed. (The old `gitlab-handbook-corpus/` folder is deprecated.)

## License

Handbook content is © GitLab, licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). This project uses
it for a personal / portfolio demo only:

- attribute GitLab and link the license (see `data/NOTICE.md`, written by the script);
- the cleaned corpus is a *modification*, so any redistribution must also be CC BY-SA 4.0;
- do not republish it as your own policy documents.

The `LICENSE` file in the cloned handbook repo is the authoritative source — verify it.

## Why "department" is not a filter

The handbook is hierarchical, not a flat set of departments. The top-level section
is kept as citation metadata (shown as `[n] title · section · score`), but the UI has
no department dropdown — retrieval is the filter.

## Adding another corpus (second product)

Do **not** dump everything into the same table without a label.

Suggested column when you get there:

```sql
ALTER TABLE documents ADD COLUMN collection TEXT NOT NULL DEFAULT 'handbook';
```

Then filter `collection = 'handbook' | 'runbook' | 'engineering-wiki'`.
See roadmap phase 2.
