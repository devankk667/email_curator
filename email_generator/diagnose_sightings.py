import sys
import pandas as pd
from cleaner import clean_email_body

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "spamassassin_emails.csv")
df["body_text"] = df["body_text"].fillna("")
if "body_html" not in df.columns:
    df["body_html"] = ""
df["body_html"] = df["body_html"].fillna("")
records = df.apply(lambda r: clean_email_body(r["body_text"], r.get("body_html", "")), axis=1)
clean_df = pd.DataFrame(list(records))
df = pd.concat([df.reset_index(drop=True), clean_df], axis=1)

mask = df["clean_text"].str.contains("sightings", case=False, na=False)
print(f"Emails containing 'sightings': {mask.sum()}")
print(df[mask]["label"].value_counts())
print()
print("Sample HAM emails containing 'sightings':")
for _, row in df[mask & (df["label"] == "ham")].head(3).iterrows():
    print(f"  {row['subject']!r}  from: {row.get('from_address', '?')}")
    print(f"    {row['clean_text'][:150]!r}")
    print()

print("Sample SPAM emails containing 'sightings' (the majority -- this is the real mystery):")
for _, row in df[mask & (df["label"] == "spam")].head(8).iterrows():
    print(f"  {row['subject']!r}  from: {row.get('from_address', '?')}")
    print(f"    {row['clean_text'][:200]!r}")
    print()