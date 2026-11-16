"""Generates the annotated NER training data for Agent 1.

Builds 60 FAKE resumes (invented people, companies, schools; example.com
emails; 555 phone numbers) in many different layouts, and records the
exact character position of every entity while the text is being written.
Because the generator places each entity itself, the labels can't drift
from the text.

Run from project root:
    python tests/fixtures/ner/make_ner_data.py

Output: tests/fixtures/ner/resumes.jsonl, one resume per line:

    {"id": "resume-07",
     "text": "Maya Okafor\\nmaya.okafor@example.com | 555-201-3344\\n...",
     "entities": [[0, 11, "PERSON", "Maya Okafor"],
                  [61, 82, "ORG", "Brightline Analytics"], ...]}

Each entity is [start, end, label, text]:
  start/end  character offsets into "text" (end is exclusive, like Python
             slicing), so text[start:end] == the entity text
  label      one of PERSON, ORG, DEGREE, JOB_TITLE, SKILL, DATE
  text       the entity text itself, only for reading/checking

Labeling guidelines:
  PERSON     the candidate's full name (header only)
  ORG        employers and schools ("Brightline Analytics", "Northfield State University")
  DEGREE     the whole degree incl. field ("Bachelor of Science in Computer Science", "MBA")
  JOB_TITLE  job titles in the work-experience section only (project roles are NOT labeled)
  SKILL      technical skills anywhere (skills list, bullets, summary); not soft skills,
             not links ("github.com/x" is not a SKILL), not "GitHub:" used as a link label
  DATE       each single date ("Jan 2022", "01/2022", "2022", "Present"); a range is
             two DATE entities, and words like "Expected" are not part of the date
  Not labeled: cities/states, course names ("Relevant Coursework: ..."), honors and
             scholarships, spoken languages ("Languages: English, Spanish"), project
             names, and roles in clubs (the club itself is an ORG)
"""
import itertools
import json
import os
import random

LABELS = ["PERSON", "ORG", "DEGREE", "JOB_TITLE", "SKILL", "DATE"]
# 60, not 30: with 30 the model often confused job titles and companies on
# unseen layouts; 45-60 resumes fixed most of that (see scripts/train_ner.py).
N_RESUMES = 60
SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
LATEST_MONTH = 2026 * 12 + 10      # Nov 2026 (months counted from 0): no job ends after this

FIRST = ["Maya", "Jonah", "Aisha", "Diego", "Lena", "Tariq", "Hana", "Marcus", "Priya",
         "Owen", "Sofia", "Kwame", "Elif", "Ravi", "Noor", "Caleb", "Ines", "Tomas",
         "Yuki", "Amara", "Felix", "Zara", "Mateo", "Leila", "Kofi", "Anya", "Sami",
         "Ivy", "Rohan", "Nadia", "Theo", "Mira"]
LAST = ["Okafor", "Lindqvist", "Rahman", "Castellanos", "Brandt", "Haddad", "Takeda",
        "Oyelaran", "Nandakumar", "Fairbanks", "Moretti", "Asante", "Yilmaz", "Iyer",
        "Siddiqui", "Whitlock", "Duarte", "Novak", "Sato", "Mensah", "Albrecht", "Qureshi",
        "Valdez", "Farouk", "Boateng", "Petrova", "Karimi", "Calloway", "Deshpande", "Ruiz"]
COMPANIES = ["Brightline Analytics", "Lakeside Robotics", "Harbor Tech Repair", "Quillstone Media",
             "Pinecrest Bakery", "Northwind Logistics", "Cobalt Ridge Software", "Silverleaf Health",
             "Redwood Data Labs", "Bluefin Systems", "Granite Peak Consulting", "Orchard Lane Grocers",
             "Tidewater Energy", "Maplewood Software", "Ironclad Payments", "Sunrise Coffee Co.",
             "Kestrel Cloud", "Hollow Oak Games", "Meridian Insurance Group", "Copperline Telecom",
             "Fernbrook Library", "Starling Retail", "Vantage Point Marketing", "Riverbend Clinic",
             "Bun-n-Bake", "Pick-n-Go Market", "Department of Public Works", "Office of Student Affairs",
             "City Water Authority"]
SCHOOLS = ["Northfield State University", "Westbrook Institute of Technology",
           "Riverbend Community College", "University of Lakemont", "Cedar Valley College",
           "Eastgate University", "Pinehurst Technical Institute", "Southport State University"]
TECH_TITLES = ["Software Engineer Intern", "Junior Data Analyst", "Web Developer", "IT Support Technician",
               "Backend Developer", "QA Tester", "Data Engineering Intern", "Machine Learning Intern",
               "Full Stack Developer", "Teaching Assistant", "Research Assistant", "Systems Administrator",
               "Frontend Developer", "Database Administrator", "DevOps Intern", "Mobile Developer",
               "Test Lead", "Software Developer (Intern)"]
OTHER_TITLES = ["Cashier", "Shift Lead", "Barista", "Sales Associate", "Library Assistant",
                "Warehouse Associate", "Customer Service Representative", "Tutor",
                "Team Member (Cashier)", "Crew Member (Drive-Thru)", "Sales Associate (Part-Time)"]
DEGREES = ["Bachelor of Science in Computer Science", "Bachelor of Science in Information Technology",
           "Bachelor of Science in Statistics", "Master of Science in Data Science",
           "Bachelor of Arts in Economics", "Associate of Science in Computer Science",
           "Associate of Applied Science in Networking", "Master of Science in Computer Engineering",
           "B.S. in Software Engineering", "BSc Computer Engineering", "M.S. in Information Systems",
           "MBA"]
# Every skill in data/skills.txt (the list Agent 3 matches job requirements
# against), so the model sees each one labeled as a SKILL at least sometimes,
# plus skills that are NOT in that list, so it also learns to find new ones.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
with open(os.path.join(ROOT, "data", "skills.txt"), encoding="utf-8") as _f:
    LISTED_SKILLS = [line.strip() for line in _f if line.strip()]
EXTRA_SKILLS = ["Kotlin", "Go", "Rust", "Swift", "Kubernetes", "PostgreSQL", "MongoDB", "Redis",
                "GraphQL", "Spring Boot", "Figma", "Jira", "Terraform", "Flutter", "Bash", "Azure",
                "Spark", "Microsoft Word"]
SKILLS = LISTED_SKILLS + EXTRA_SKILLS
PROJECTS = ["Campus Event Finder", "Budget Buddy", "Sales Trend Dashboard", "Line-Following Robot",
            "Library Checkout Analyzer", "Recipe Recommender", "Chess Engine", "Weather Alert Bot",
            "Parking Spot Tracker", "Study Group Matcher", "Inventory Forecaster", "Trail Map App"]
PROJECT_ROLES = ["Team Lead", "Developer", "Solo Project", "Backend Lead", "Contributor"]
CITIES = ["Monroe, LA", "Austin, TX", "Remote", "Dallas, TX", "Denver, CO", "Raleigh, NC"]
CITY_STATES = ["Monroe, Louisiana", "Austin, Texas", "Raleigh, North Carolina", "Denver, Colorado",
               "Tacoma, Washington", "Dayton, Ohio"]
COURSES = ["Data Structures", "Operating Systems", "Computer Networks", "Discrete Mathematics",
           "Database Systems", "Analysis of Algorithms", "Software Engineering", "Linear Algebra",
           "Calculus II", "Computer Architecture"]
HONORS = ["Dean's List", "Presidential Scholarship", "Merit Scholarship", "Honors College",
          "President's List"]
SPOKEN = ["English", "Spanish", "French", "Mandarin", "Hindi", "Bengali", "Arabic", "Portuguese"]
CLUBS = ["Robotics Club", "Campus Coding Club", "Women in Computing Society", "Habitat Volunteers",
         "Math Tutoring Center", "Student Government Association"]
CLUB_ROLES = ["Vice President", "Treasurer", "Volunteer Tutor", "Event Coordinator", "Member"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
BULLETS = ["• ", "- ", "● ", "* ", "▪ "]
VERBS = ["Built", "Developed", "Maintained", "Automated", "Designed", "Tested", "Improved", "Wrote"]
THINGS = ["internal dashboards", "a customer portal", "data pipelines", "REST endpoints",
          "monitoring scripts", "a scheduling tool", "unit tests", "reporting jobs"]
NON_TECH_BULLETS = ["Handled customer payments and returns", "Trained new team members",
                    "Managed opening and closing shifts", "Kept inventory records up to date",
                    "Resolved customer questions in person and by phone"]


class Builder:
    """Writes text and remembers where each labeled piece starts and ends."""

    def __init__(self):
        self.text = ""
        self.entities = []

    def add(self, piece, label=None):
        if label:
            self.entities.append([len(self.text), len(self.text) + len(piece), label, piece])
        self.text += piece

    def newline(self):
        self.text += "\n"


class Resume:
    def __init__(self, rng, index):
        self.rng = rng
        self.b = Builder()
        self.index = index
        self.date_style = rng.choice(["short", "long", "numeric", "year", "short_dot"])
        self.separator = rng.choice([" - ", " – ", " to ", "-"])
        self.heading_style = rng.choice(["upper", "title", "colon"])
        self.bullet = rng.choice(BULLETS)

    # ---------- small pieces ----------

    def date_text(self, year, month):
        style = self.date_style
        if style == "short":
            return f"{MONTHS[month - 1][:3]} {year}"
        if style == "short_dot":
            short = MONTHS[month - 1][:3]
            return f"{short}. {year}" if short != "May" else f"May {year}"
        if style == "long":
            return f"{MONTHS[month - 1]} {year}"
        if style == "numeric":
            return f"{month:02d}/{year}"
        return str(year)

    def add_range(self, start, end):
        self.b.add(self.date_text(*start), "DATE")
        self.b.add(self.separator)
        if end is None:
            self.b.add(self.rng.choice(["Present", "Current"]), "DATE")
        else:
            self.b.add(self.date_text(*end), "DATE")

    def heading(self, words):
        word = self.rng.choice(words)
        if self.heading_style == "upper":
            word = word.upper()
        elif self.heading_style == "colon":
            word += ":"
        self.b.add(word)
        self.b.newline()

    def skill_sentence(self, skills):
        # A few sentence shapes, e.g. "Built data pipelines with Python and SQL",
        # "Applied OOP principles to a scheduling tool", "Used Git and GitHub for ..."
        rng, b = self.rng, self.b
        shape = rng.randint(1, 3)
        if shape == 1:
            b.add(f"{rng.choice(VERBS)} {rng.choice(THINGS)} with ")
        elif shape == 2:
            b.add("Applied ")
        else:
            b.add("Used ")
        for i, skill in enumerate(skills):
            if i:
                b.add(" and " if i == len(skills) - 1 else ", ")
            b.add(skill, "SKILL")
        if shape == 2:
            b.add(f" principles to {rng.choice(THINGS)}")
        elif shape == 3:
            b.add(f" to deliver {rng.choice(THINGS)}")

    # ---------- sections ----------

    def header(self, name):
        rng, b = self.rng, self.b
        first, last = name
        email = f"{first.lower()}.{last.lower()}@example.com"
        phone = rng.choice([f"555-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}",
                            f"(555) {rng.randint(100, 999)}-{rng.randint(1000, 9999)}",
                            f"555.{rng.randint(100, 999)}.{rng.randint(1000, 9999)}"])
        link = rng.choice([f"linkedin.com/in/{first.lower()}-{last.lower()}",
                           f"github.com/{first.lower()}{last.lower()}"])
        style = rng.randint(1, 5)
        full = f"{first} {last}"
        if style == 4:
            full = full.upper()
        b.add(full, "PERSON")
        b.newline()
        if style == 1:
            b.add(f"{email} | {phone} | {link}")
        elif style == 2:
            b.add(f"{rng.choice(CITIES)}\n{email}\n{phone}")
        elif style == 3:
            label = "GitHub" if "github" in link else "LinkedIn"
            b.add(f"Email: {email}  Phone: {phone}\n{label}: {link}")
        elif style == 4:
            b.add(f"{phone} · {email}")
        else:
            b.add(f"{rng.choice(CITY_STATES)} | {email} | {phone}\n"
                  f"LinkedIn: linkedin.com/in/{first.lower()}{rng.randint(1, 99)}")
        b.newline()

    def summary(self, skills):
        if self.rng.random() < 0.5:
            return
        self.heading(["Summary", "Profile", "Objective"])
        self.b.add(self.rng.choice(["Motivated student", "Detail-oriented developer",
                                    "Early-career analyst"]) + " experienced in ")
        self.b.add(skills[0], "SKILL")
        self.b.add(" and ")
        self.b.add(skills[1], "SKILL")
        self.b.add(", looking for an entry-level role.")
        self.b.newline()

    def job(self, company, title, start, end, tech, skills):
        rng, b = self.rng, self.b
        layout = rng.randint(1, 7)
        if layout == 1:                      # Company | Title \n dates
            b.add(company, "ORG"); b.add(" | "); b.add(title, "JOB_TITLE"); b.newline()
            self.add_range(start, end)
        elif layout == 2:                    # Title, Company \n dates
            b.add(title, "JOB_TITLE"); b.add(", "); b.add(company, "ORG"); b.newline()
            self.add_range(start, end)
        elif layout == 3:                    # Title at Company (dates)
            b.add(title, "JOB_TITLE"); b.add(" at "); b.add(company, "ORG"); b.add(" (")
            self.add_range(start, end); b.add(")")
        elif layout == 4:                    # Title \n Company, City \n dates
            b.add(title, "JOB_TITLE"); b.newline()
            b.add(company, "ORG"); b.add(f", {rng.choice(CITIES)}"); b.newline()
            self.add_range(start, end)
        elif layout == 5:                    # Company — Title — dates
            b.add(company, "ORG"); b.add(" — "); b.add(title, "JOB_TITLE"); b.add(" — ")
            self.add_range(start, end)
        elif layout == 6:                    # Title | Company | dates
            b.add(title, "JOB_TITLE"); b.add(" | "); b.add(company, "ORG"); b.add(" | ")
            self.add_range(start, end)
        else:                                # Company | Title \n City, State dates
            b.add(company, "ORG"); b.add(" | "); b.add(title, "JOB_TITLE"); b.newline()
            b.add(f"{rng.choice(CITY_STATES)} ")
            self.add_range(start, end)
        b.newline()
        for _ in range(rng.randint(1, 2)):
            b.add(self.bullet)
            if tech:
                self.skill_sentence(rng.sample(skills, rng.randint(1, 2)))
            else:
                b.add(rng.choice(NON_TECH_BULLETS))
            b.newline()

    def experience(self, skills):
        rng = self.rng
        self.heading(["Experience", "Work Experience", "Professional Experience",
                      "Employment History", "Relevant Experience"])
        companies = rng.sample(COMPANIES, rng.randint(1, 3))
        year = rng.randint(2023, 2026)
        for i, company in enumerate(companies):
            tech = rng.random() < 0.7
            title = rng.choice(TECH_TITLES if tech else OTHER_TITLES)
            length = rng.randint(3, 30)                    # months
            end_m = min(year * 12 + rng.randint(0, 11), LATEST_MONTH)
            start_m = end_m - length
            end = None if i == 0 and rng.random() < 0.5 else (end_m // 12, end_m % 12 + 1)
            self.job(company, title, (start_m // 12, start_m % 12 + 1), end, tech, skills)
            year = start_m // 12 - rng.randint(0, 1)

    def education(self):
        rng, b = self.rng, self.b
        self.heading(["Education", "Academic Background"])
        school, degree = rng.choice(SCHOOLS), rng.choice(DEGREES)
        grad = (rng.randint(2019, 2029), rng.choice([5, 8, 12]))
        layout = rng.randint(1, 5)
        if layout == 3 and grad[0] < 2027:   # "Expected" only makes sense for a future date
            layout = 1
        if layout == 1:                      # School \n Degree \n date
            b.add(school, "ORG"); b.newline(); b.add(degree, "DEGREE"); b.newline()
            b.add(self.date_text(*grad), "DATE")
        elif layout == 2:                    # Degree, School, date
            b.add(degree, "DEGREE"); b.add(", "); b.add(school, "ORG"); b.add(", ")
            b.add(self.date_text(*grad), "DATE")
        elif layout == 3:                    # School — Degree (Expected date)
            b.add(school, "ORG"); b.add(" — "); b.add(degree, "DEGREE"); b.add(" (Expected ")
            b.add(self.date_text(*grad), "DATE"); b.add(")")
        elif layout == 4:                    # Degree | School | GPA | date
            b.add(degree, "DEGREE"); b.add(" | "); b.add(school, "ORG")
            b.add(f" | GPA {rng.choice(['3.2', '3.5', '3.8', '3.9'])} | ")
            b.add(self.date_text(*grad), "DATE")
        else:                                # School \n City, State date \n Degree
            b.add(school, "ORG"); b.newline()
            b.add(f"{rng.choice(CITY_STATES)} "); b.add(self.date_text(*grad), "DATE"); b.newline()
            b.add(degree, "DEGREE")
        b.newline()
        # Extra lines that are NOT entities: honors and coursework
        if rng.random() < 0.5:
            b.add(f"{self.bullet}Honors: " + "; ".join(rng.sample(HONORS, 2))); b.newline()
        if rng.random() < 0.6:
            b.add(f"{self.bullet}Relevant Coursework: " + ", ".join(rng.sample(COURSES, 4)))
            b.newline()

    def skills_section(self, skills):
        rng, b = self.rng, self.b
        self.heading(["Skills", "Technical Skills", "Core Competencies", "Tools & Technologies"])
        style = rng.randint(1, 4)
        if style == 1:                       # comma-separated line
            for i, skill in enumerate(skills):
                if i:
                    b.add(", ")
                b.add(skill, "SKILL")
            b.newline()
        elif style == 2:                     # one bullet per skill
            for skill in skills:
                b.add(self.bullet); b.add(skill, "SKILL"); b.newline()
        else:                                # categories (style 4: bulleted, with a period)
            half = len(skills) // 2
            names = rng.choice([("Languages", "Tools"), ("Programming", "Tools & Software"),
                                ("Programming Languages", "Frameworks")])
            prefix, end = (self.bullet, ".") if style == 4 else ("", "")
            for category, group in zip(names, [skills[:half], skills[half:]]):
                b.add(f"{prefix}{category}: ")
                for i, skill in enumerate(group):
                    if i:
                        b.add(", ")
                    b.add(skill, "SKILL")
                # No final "." after a one-letter skill: spaCy reads "R." as one token
                b.add(end if len(group[-1]) > 1 else "")
                b.newline()
        if rng.random() < 0.4:               # spoken languages are NOT skills
            b.add(f"{'Spoken Languages' if style == 3 and names[0] == 'Languages' else 'Languages'}: "
                  + ", ".join(rng.sample(SPOKEN, 3)))
            b.newline()

    def projects(self, skills):
        rng, b = self.rng, self.b
        if rng.random() < 0.3:
            return
        self.heading(["Projects", "Academic Projects", "Personal Projects"])
        for name in rng.sample(PROJECTS, rng.randint(1, 2)):
            b.add(name)                           # project names are not entities
            if rng.random() < 0.6:
                b.add(f" | {rng.choice(PROJECT_ROLES)}")   # project roles are not JOB_TITLE
            if rng.random() < 0.6:
                b.add(" | ")
                b.add(self.date_text(rng.randint(2021, 2026), rng.randint(1, 12)), "DATE")
            b.newline()
            b.add(self.bullet)
            self.skill_sentence(rng.sample(skills, rng.randint(1, 3)))
            b.newline()

    def leadership(self):
        rng, b = self.rng, self.b
        if rng.random() < 0.6:
            return
        self.heading(["Leadership", "Achievements and Leadership", "Activities"])
        for club in rng.sample(CLUBS, rng.randint(1, 2)):
            b.add(f"{rng.choice(CLUB_ROLES)} | ")            # club roles are not JOB_TITLE
            b.add(club, "ORG"); b.add(" ")
            year = rng.randint(2022, 2026)
            self.add_range((year, rng.randint(1, 10)), None)
            b.newline()
            b.add(f"{self.bullet}Organized weekly meetings and events for members")
            b.newline()

    def build(self, name):
        skills = self.rng.sample(SKILLS, self.rng.randint(5, 10))
        self.header(name)
        self.summary(skills)
        sections = [lambda: self.experience(skills), self.education,
                    lambda: self.skills_section(skills), lambda: self.projects(skills),
                    self.leadership]
        if self.rng.random() < 0.5:                       # education first for some students
            sections[0], sections[1] = sections[1], sections[0]
        for section in sections:
            section()
        return {"id": f"resume-{self.index:02d}", "text": self.b.text.rstrip("\n"),
                "entities": self.b.entities}


def generate(n=N_RESUMES, seed=SEED):
    rng = random.Random(seed)
    names = rng.sample(list(itertools.product(FIRST, LAST)), n)     # unique full names
    return [Resume(rng, i + 1).build(names[i]) for i in range(n)]


def check_alignment(records):
    """Every span must start and end on a token boundary of spaCy's English
    tokenizer, otherwise spaCy can't train on it. Returns a list of problems."""
    import spacy
    tokenizer = spacy.blank("en")
    problems = []
    for r in records:
        doc = tokenizer.make_doc(r["text"])
        for start, end, label, text in r["entities"]:
            if r["text"][start:end] != text:
                problems.append(f"{r['id']}: offsets don't match text {text!r}")
            elif doc.char_span(start, end, label=label) is None:
                problems.append(f"{r['id']}: {label} {text!r} not on token boundaries")
    return problems


def save(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    records = generate()
    problems = check_alignment(records)
    if problems:
        raise SystemExit("\n".join(problems))
    out = os.path.join(HERE, "resumes.jsonl")
    save(records, out)
    counts = {label: sum(e[2] == label for r in records for e in r["entities"]) for label in LABELS}
    print(f"wrote {len(records)} resumes to {out}")
    print("entities per label:", counts, "total:", sum(counts.values()))
