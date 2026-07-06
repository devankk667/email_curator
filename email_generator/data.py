"""
Static data pools used to fill in templates: names, domains, companies,
cities, roles, courses, events, signatures.
"""

FIRST_NAMES = [
    "Rahul", "Ananya", "Wei", "Fatima", "Diego", "Priya", "James", "Mei",
    "Carlos", "Sara", "Arjun", "Elena", "Yuki", "Omar", "Nina", "Liam",
    "Zara", "Kenji", "Isabella", "Noah", "Aisha", "Lucas", "Meera", "Ivan",
]

LAST_NAMES = [
    "Sharma", "Patel", "Chen", "Khan", "Garcia", "Nguyen", "Smith", "Kim",
    "Rossi", "Okafor", "Ivanov", "Silva", "Lopez", "Tanaka", "Mehta",
    "Dubois", "Novak", "Fischer", "Costa", "Reddy",
]

COMPANIES = [
    "Zynta Corp", "Nimbus Retail", "PixelWorks", "Orbit Logistics",
    "Vertex Bank", "Cloudscape", "Quanta Labs", "Fresco Foods",
    "Brightline Media", "Fernwood Analytics", "Starforge Games",
    "Coral Health", "Meridian Airlines", "Anchorpoint Realty",
]

CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Singapore", "London", "New York",
    "Toronto", "Dubai", "Tokyo", "Berlin", "Sydney", "Amsterdam",
    "San Francisco", "Chennai", "Paris",
]

ROLES = [
    "Software Engineer", "Data Analyst", "Product Manager", "ML Engineer",
    "Backend Developer", "UX Designer", "DevOps Engineer",
    "Business Analyst", "QA Engineer", "Data Scientist",
]

COURSES = [
    "Operating Systems", "Data Structures", "Database Systems",
    "Computer Networks", "Linear Algebra", "Machine Learning",
    "Software Engineering", "Discrete Mathematics",
]

TOPICS = [
    "Machine Learning", "Cloud Computing", "Cybersecurity",
    "Distributed Systems", "Generative AI", "Blockchain",
]

BOOKS = [
    "Introduction to Algorithms", "Clean Code", "Design Patterns",
    "The Pragmatic Programmer", "Deep Learning", "Database System Concepts",
]

EVENTS = [
    "Sarah's housewarming", "the team offsite", "the college reunion",
    "the birthday party", "the hackathon kickoff", "the alumni meetup",
]

# URL pools used to fill {url} placeholders in templates. Split into
# "legit-looking" (matches the sending category/domain) and "suspicious"
# (used only in Spam templates) so has_url / url_count actually carry
# signal for a spam classifier, not just random noise.
# URL pools used to fill {url} placeholders in templates, split per
# category so a shopping tracking link actually looks like a shopping
# domain rather than a random unrelated one. SUSPICIOUS_URLS is used
# only for Spam templates.
URLS_BY_CATEGORY = {
    "College": [
        "https://portal.university.edu/exams",
        "https://portal.university.edu/results",
        "https://college.ac.in/scholarships",
        "https://campusmail.edu/library",
    ],
    "Finance": [
        "https://vertexbank.com/statements",
        "https://vertexbank.com/accounts",
        "https://payflow.com/dashboard",
        "https://vertexbank.com/cards/offers",
    ],
    "Shopping": [
        "https://nimbusretail.com/orders/track",
        "https://amazon.in/orders",
        "https://flipkart.com/returns",
        "https://myntra.com/orders",
    ],
    "Travel": [
        "https://meridianair.com/checkin",
        "https://meridianair.com/bookings",
        "https://makemytrip.com/bookings",
        "https://bookings.com/reservations",
    ],
    "Social": [
        "https://facebookmail.com/profile",
        "www.linkedin.com/in/connect",
        "https://instagram.com/notifications",
        "https://meetup.com/groups",
    ],
    "Job": [
        "https://greenhouse.io/applications",
        "www.linkedin.com/jobs/view",
        "https://workday.com/candidate",
        "https://naukri.com/jobs",
    ],
    "Government": [
        "https://gov.in/services/passport",
        "https://passportseva.gov.in/status",
        "https://municipal.gov/elections",
        "https://gov.in/services/rto",
    ],
}

SUSPICIOUS_URLS = [
    "https://claim-now.biz/prize",
    "https://luckywin-prize.com/verify",
    "https://freegift-alert.net/redeem",
    "https://cashbonus.xyz/claim",
    "http://secure-verify-account.com/login",
    "http://account-update-now.info/confirm",
    "https://bit.ly/3xF9kLp",
    "http://192.168.44.12/claim",
]

# Domains grouped by category so senders look plausible for the mail type.
DOMAINS_BY_CATEGORY = {
    "College": ["university.edu", "college.ac.in", "campusmail.edu"],
    "Finance": ["vertexbank.com", "payflow.com", "taxportal.gov.in", "citibank.com"],
    "Shopping": ["nimbusretail.com", "amazon.in", "flipkart.com", "myntra.com"],
    "Travel": ["meridianair.com", "bookings.com", "makemytrip.com", "airindia.com"],
    "Spam": ["luckywin-prize.com", "claim-now.biz", "freegift-alert.net", "cashbonus.xyz"],
    "Social": ["facebookmail.com", "linkedin.com", "instagram.com", "meetup.com"],
    "Job": ["greenhouse.io", "linkedin.com", "workday.com", "naukri.com"],
    "Government": ["gov.in", "irs.gov", "passportseva.gov.in", "municipal.gov"],
}

# Short sign-offs appended to plain-text bodies, per category.
SIGNATURES_BY_CATEGORY = {
    "College": ["\n\nRegards,\nOffice of Academic Affairs", "\n\nThank you,\nExamination Cell"],
    "Finance": ["\n\nRegards,\nCustomer Service Team", "\n\nThank you for banking with us."],
    "Shopping": ["\n\nHappy Shopping!\nThe Team", "\n\nThanks for being a valued customer."],
    "Travel": ["\n\nSafe travels,\nBooking Support Team", "\n\nRegards,\nReservations Desk"],
    "Spam": ["\n\nAct now, offer expires soon!", "\n\nDon't miss this one-time opportunity!"],
    "Social": ["\n\n-- Sent from your notifications", ""],
    "Job": ["\n\nBest regards,\nTalent Acquisition Team", "\n\nWe look forward to hearing from you."],
    "Government": ["\n\nRegards,\nPublic Services Department", "\n\nThis is an automated notice."],
}
