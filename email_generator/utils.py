"""
Small random-generation helpers shared across the project.
"""

import random
import uuid
from datetime import datetime, timedelta
from email.utils import format_datetime

import data


def random_date_between(start_str, end_str):
    """Return a random datetime between two 'YYYY-MM-DD' strings."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    delta_days = (end - start).days
    if delta_days <= 0:
        return start
    offset = random.randint(0, delta_days)
    hour = random.randint(6, 22)
    minute = random.choice([0, 5, 10, 15, 20, 30, 45])
    return start + timedelta(days=offset, hours=hour, minutes=minute)


def rfc2822_date(dt):
    """Format a datetime object as an RFC 2822 email Date header."""
    return format_datetime(dt)


def human_date(dt, fmt="%B %d"):
    return dt.strftime(fmt)


def random_name():
    return f"{random.choice(data.FIRST_NAMES)} {random.choice(data.LAST_NAMES)}"


def random_email_address(name, category):
    domain = random.choice(data.DOMAINS_BY_CATEGORY[category])
    local = name.lower().replace(" ", ".")
    # occasionally add a number suffix for realism
    if random.random() < 0.3:
        local += str(random.randint(1, 99))
    return f"{local}@{domain}"

def random_recipient_address():
    name = random_name()
    local = name.lower().replace(" ", ".")
    return f"{local}@myinbox.com"


def random_amount():
    return random.choice([199, 499, 999, 1499, 2500, 3999, 7500, 12000, 25000, 45000])


def random_percent():
    return round(random.uniform(1.5, 22.0), 1)


def make_message_id(domain):
    return f"<{uuid.uuid4().hex}@{domain}>"


def maybe(prob):
    return random.random() < prob


def wrap_html(subject, body_text, signature, category=None):
    """Very small HTML wrapper so we get a plausible multipart/alternative part.
    Note: category is intentionally NOT rendered anywhere in the output --
    the label must never leak into the email content itself, only into
    labels.csv, or a classifier trained on this data would be able to
    'cheat' by reading it instead of learning real patterns."""
    paragraphs = "".join(f"<p>{line}</p>" for line in body_text.strip().split("\n") if line.strip())
    sig_html = signature.replace("\n", "<br>")
    return f"""<html>
  <body style="font-family: Arial, sans-serif; color:#222;">
    <h3 style="color:#444;">{subject}</h3>
    {paragraphs}
    <p style="color:#888; font-size:12px;">{sig_html}</p>
  </body>
</html>"""


def fill_placeholders(template, category):
    """Fill a {placeholder} template string with randomized values."""
    dt = random_date_between("2026-01-01", "2026-12-31")
    dt2 = random_date_between("2026-01-01", "2026-12-31")
    url_pool = data.SUSPICIOUS_URLS if category == "Spam" else data.URLS_BY_CATEGORY.get(category, data.SUSPICIOUS_URLS)
    return template.format(
        name=random_name(),
        date=human_date(dt),
        date2=human_date(dt2),
        amount=random_amount(),
        amount2=random_amount(),
        company=random.choice(data.COMPANIES),
        city=random.choice(data.CITIES),
        course=random.choice(data.COURSES),
        topic=random.choice(data.TOPICS),
        book=random.choice(data.BOOKS),
        event=random.choice(data.EVENTS),
        role=random.choice(data.ROLES),
        num=random.randint(1, 9),
        room=random.randint(100, 499),
        pct=random_percent(),
        ref=str(random.randint(100000, 999999)),
        url=random.choice(url_pool),
    )
