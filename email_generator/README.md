# email_generator

Generates a synthetic inbox of realistic `.eml` files for the mail-ai
project's Phase 1 (and beyond — the metadata is rich enough for NER,
priority prediction, summarization, and semantic search too).

## Structure

```
email_generator/
├── config.py            # counts, weights, probabilities — edit and re-run
├── data.py               # names, domains, companies, cities, roles, signatures
├── utils.py               # random helpers (dates, addresses, HTML wrapping)
├── templates.py            # 96 subject/body templates across 8 categories
├── generator.py             # main script — run this
├── labels.csv                # filename -> category, sender, date, subject, thread/attachment flags
├── metadata.json               # dataset-level summary
└── generated_emails/
    └── email_00001.eml ... email_10000.eml
```

## Run

```
python generator.py
```

Re-running overwrites `generated_emails/`. Change `RANDOM_SEED = None` in
`config.py` for a different random batch each run, or set an int for
reproducibility.

## What's in each .eml

- Standard headers: From, To, Subject, Date (RFC 2822), Message-ID
- multipart/alternative body: text/plain + text/html
- ~15% are threaded replies/forwards (Re:/Fwd: subject, In-Reply-To /
  References headers pointing at a real prior Message-ID, quoted original
  text)
- ~12% carry a small text attachment (invoice, boarding pass, job
  description, etc. depending on category)

## Categories (weighted, not uniform — see config.py to change)

Shopping (22%), Spam (18%), Social (15%), Finance (12%), College (10%),
Job (10%), Travel (8%), Government (5%)

## labels.csv columns

| column | meaning |
|---|---|
| `filename` | which `.eml` file |
| `category` | ground-truth category (target for Phase 4 classifier) |
| `sender`, `date`, `subject` | convenience copies of header fields |
| `is_thread_reply` | True if this is a Re:/Fwd: reply |
| `has_attachment` | True if a dummy attachment was added |
| `spam_score` | 0-100, heuristic (keyword + category based) — target for a spam-scoring model |
| `priority_score` | 0-100, heuristic (keyword + category based) — the underlying continuous score |
| `priority` | Low/Medium/High bucket of `priority_score` — target for Phase 8 priority prediction |

`spam_score` and `priority` are computed in `labeling.py` from the actual
subject/body text (urgency words, deadline phrasing, prize/verify-account
language, etc.) plus a per-category base rate and random jitter — not a
straight lookup from `category`. That's intentional: a model trained on
these labels has to learn real textual patterns, the same way it will
have to on real mail where there's no category to peek at. As with
`category`, these labels are **only** in `labels.csv`, never inside the
`.eml` content.



Python's standard library parses these directly:

```python
from email import policy
from email.parser import BytesParser

with open("generated_emails/email_00001.eml", "rb") as f:
    msg = BytesParser(policy=policy.default).parse(f)

print(msg["subject"], msg["from"], msg["date"])
print(msg.get_body(preferencelist=("plain",)).get_content())
```

This mirrors how you'll eventually parse real Gmail API message payloads,
so your Phase 2 cleaning/parsing code should carry over with minimal
changes.
