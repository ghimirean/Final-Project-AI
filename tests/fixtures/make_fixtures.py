"""Builds the FAKE test resumes in tests/fixtures/.

All names, emails, phones, and companies are invented. Run from project root:
    python tests/fixtures/make_fixtures.py
"""
import os
import docx

HERE = os.path.dirname(os.path.abspath(__file__))

RESUMES = {
    # Style 1: Title Case headings, normal "Mon YYYY - Mon YYYY" dates
    "resume_title_case.docx": [
        "Priya Nandakumar",
        "priya.nandakumar@example.com | (555) 201-3344 | linkedin.com/in/priya-nandakumar-fake",
        "Professional Summary",
        "Data analyst who turns messy spreadsheets into clear dashboards.",
        "Work Experience",
        "Brightline Analytics | Junior Data Analyst",
        "Jan 2022 - Dec 2023",
        "• Built weekly sales dashboards in Tableau",
        "Cedar Grove Library | Student Assistant",
        "Sep 2020 - Dec 2021",
        "• Organized digital catalog records in Excel",
        "Technical Skills",
        "• Python",
        "• SQL",
        "• Tableau",
        "• pandas",
        "Education",
        "Northfield State University",
        "Bachelor of Science in Statistics",
        "May 2022",
    ],
    # Style 2: numeric "MM/YYYY - MM/YYYY" dates
    "resume_numeric_dates.docx": [
        "Marcus Oyelaran",
        "marcus.oyelaran@example.org",
        "555-404-9821",
        "github.com/marcus-oyelaran-fake",
        "EXPERIENCE",
        "Lakeside Robotics | Software Engineer Intern",
        "01/2023 - 05/2024",
        "- Wrote control code in C++ and tested it on Linux",
        "Harbor Tech Repair | IT Support Technician",
        "06/2021 - 12/2022",
        "- Fixed laptops and set up Git for the team",
        "SKILLS",
        "- C++",
        "- Java",
        "- Linux",
        "- Docker",
        "EDUCATION",
        "Westbrook Institute of Technology",
        "Bachelor of Science in Computer Engineering",
        "Dec 2024",
    ],
    # Style 3: year-only dates ("2021 - Present"), comma-separated skills
    "resume_comma_skills.docx": [
        "Elena Vasquez-Moreau",
        "Email: elena.vm@example.net  Phone: 555.310.7788",
        "Work Experience",
        "Quillstone Media | Web Developer",
        "2021 - Present",
        "Built client sites with React and Node.js.",
        "Pinecrest Bakery | Shift Lead",
        "2018 - 2021",
        "Managed schedules for a team of six.",
        "Skills",
        "JavaScript, React, Node.js, HTML, CSS, Git, REST API",
        "Education",
        "Riverbend Community College",
        "Associate of Science in Computer Science",
        "May 2018",
    ],
}


def main():
    for filename, lines in RESUMES.items():
        document = docx.Document()
        for line in lines:
            document.add_paragraph(line)
        path = os.path.join(HERE, filename)
        document.save(path)
        print("wrote", path)


if __name__ == "__main__":
    main()
