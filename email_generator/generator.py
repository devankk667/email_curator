"""
Main entry point. Generates a synthetic inbox of realistic .eml files
(with headers, multipart text+HTML bodies, occasional attachments and
Re:/Fwd: threads), plus labels.csv and metadata.json.

Run:
    python generator.py
"""

import csv
import json
import random
from collections import defaultdict
from email.message import EmailMessage
from pathlib import Path

import config
import data
import heuristic_labels
import templates
import utils

if config.RANDOM_SEED is not None:
    random.seed(config.RANDOM_SEED)

OUTPUT_DIR = Path(config.OUTPUT_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)

INBOX_OWNER = "you@myinbox.com"

# Keeps recent (message_id, subject, sender, body_text) per category so we
# can build plausible Re:/Fwd: threads that reference a real prior message.
recent_by_category = defaultdict(list)
MAX_RECENT = 40


def category_counts():
    total_weight = sum(config.CATEGORY_WEIGHTS.values())
    counts = {}
    assigned = 0
    categories = list(config.CATEGORY_WEIGHTS.items())
    for i, (cat, weight) in enumerate(categories):
        if i == len(categories) - 1:
            counts[cat] = config.TOTAL_EMAILS - assigned  # remainder to last category
        else:
            n = round(config.TOTAL_EMAILS * weight / total_weight)
            counts[cat] = n
            assigned += n
    return counts


def build_email(category, index):
    subj_t, body_t = random.choice(templates.TEMPLATES[category])
    subject = utils.fill_placeholders(subj_t, category)
    body = utils.fill_placeholders(body_t, category)
    signature = random.choice(data.SIGNATURES_BY_CATEGORY[category])

    sender_name = utils.random_name()
    if category in ("Job", "Social"):
        # occasionally send from a company/platform name instead of a person
        if utils.maybe(0.4):
            sender_name = random.choice(data.COMPANIES) if category == "Job" else random.choice(
                ["Facebook", "LinkedIn", "Instagram", "Meetup"]
            )
    elif category in ("Shopping", "Finance", "Travel", "Government"):
        sender_name = random.choice(data.COMPANIES) if category != "Government" else "Government Services"

    sender_addr = utils.random_email_address(sender_name.split()[0], category)
    dt = utils.random_date_between(config.DATE_RANGE_START, config.DATE_RANGE_END)
    msg_id = utils.make_message_id(sender_addr.split("@")[1])

    is_thread = False
    in_reply_to = None
    references = None
    thread_root_id = msg_id

    if recent_by_category[category] and utils.maybe(config.THREAD_PROBABILITY):
        prior_id, prior_subject, prior_sender, prior_body = random.choice(recent_by_category[category])
        prefix = random.choice(templates.THREAD_PREFIXES)
        subject = prefix + prior_subject
        snippet = random.choice(templates.REPLY_SNIPPETS)
        quoted = "\n".join(f"> {line}" for line in prior_body.strip().split("\n"))
        body = f"{snippet}\n\n{quoted}"
        in_reply_to = prior_id
        references = prior_id
        thread_root_id = prior_id
        is_thread = True

    full_body = body + signature
    html_body = utils.wrap_html(subject.replace(templates.THREAD_PREFIXES[0], "").replace(
        templates.THREAD_PREFIXES[1], ""), body, signature, category)

    msg = EmailMessage()
    msg["From"] = f"{sender_name} <{sender_addr}>"
    msg["To"] = INBOX_OWNER
    msg["Subject"] = subject
    msg["Date"] = utils.rfc2822_date(dt)
    msg["Message-ID"] = msg_id
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
        msg["References"] = references

    msg.set_content(full_body)
    msg.add_alternative(html_body, subtype="html")

    has_attachment = utils.maybe(config.ATTACHMENT_PROBABILITY)
    if has_attachment:
        attach_name, attach_text = make_attachment(category)
        msg.add_attachment(
            attach_text.encode("utf-8"),
            maintype="text",
            subtype="plain",
            filename=attach_name,
        )

    # remember this email so future emails in the same category can reply to it
    recent_by_category[category].append((msg_id, subject.split(": ", 1)[-1] if is_thread else subject, sender_addr, body))
    if len(recent_by_category[category]) > MAX_RECENT:
        recent_by_category[category].pop(0)

    spam_score, priority_score, priority = heuristic_labels.compute_labels(category, subject, body)

    return {
        "msg": msg,
        "category": category,
        "sender": sender_addr,
        "date": dt.isoformat(),
        "subject": subject,
        "is_thread": is_thread,
        "thread_root_id": thread_root_id,
        "has_attachment": has_attachment,
        "spam_score": spam_score,
        "priority_score": priority_score,
        "priority": priority,
    }


def make_attachment(category):
    names = {
        "College": ("assignment_brief.txt", "Assignment brief:\nComplete all sections and submit via the portal."),
        "Finance": ("statement.txt", "Statement summary:\nSee attached transaction details for this period."),
        "Shopping": ("invoice.txt", "Invoice:\nThank you for your purchase. See item breakdown attached."),
        "Travel": ("boarding_pass.txt", "Boarding Pass:\nSeat 14C, Gate 22, Boarding starts 30 min prior."),
        "Spam": ("claim_form.txt", "Fill this form immediately to claim your prize."),
        "Social": ("photo_details.txt", "Photo details and tag information."),
        "Job": ("job_description.txt", "Full job description and requirements attached."),
        "Government": ("notice.txt", "Official notice document. Please retain for your records."),
    }
    return names.get(category, ("attachment.txt", "See attached details."))


def main():
    counts = category_counts()
    rows = []
    total_written = 0

    for category, n in counts.items():
        for i in range(n):
            record = build_email(category, i)
            filename = f"email_{total_written + 1:05d}.eml"
            filepath = OUTPUT_DIR / filename
            with open(filepath, "wb") as f:
                f.write(bytes(record["msg"]))

            rows.append({
                "filename": filename,
                "category": record["category"],
                "sender": record["sender"],
                "date": record["date"],
                "subject": record["subject"],
                "is_thread_reply": record["is_thread"],
                "has_attachment": record["has_attachment"],
                "spam_score": record["spam_score"],
                "priority_score": record["priority_score"],
                "priority": record["priority"],
            })
            total_written += 1

    # labels.csv
    labels_path = OUTPUT_DIR.parent / "labels.csv"
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "category", "sender", "date", "subject",
            "is_thread_reply", "has_attachment",
            "spam_score", "priority_score", "priority",
        ])
        writer.writeheader()
        writer.writerows(rows)

    # metadata.json
    meta = {
        "total_emails": total_written,
        "categories": counts,
        "date_range": [config.DATE_RANGE_START, config.DATE_RANGE_END],
        "attachment_probability": config.ATTACHMENT_PROBABILITY,
        "thread_probability": config.THREAD_PROBABILITY,
        "random_seed": config.RANDOM_SEED,
        "format": "RFC 5322 .eml files, multipart/mixed with text/plain + text/html alternative parts, optional text attachment",
    }
    meta_path = OUTPUT_DIR.parent / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Generated {total_written} .eml files in {OUTPUT_DIR}/")
    print(f"Labels written to {labels_path}")
    print(f"Metadata written to {meta_path}")
    for cat, n in counts.items():
        print(f"  {cat:12s}: {n}")


if __name__ == "__main__":
    main()
