#!/usr/bin/env python3
"""
Script to initialize the PostgreSQL database and seed it with sample data.
"""
import random
import uuid
from datetime import datetime, timedelta
from faker import Faker
from sqlalchemy.orm import Session
from db.models import Job, Applicant, ApplicantSkill, ApplicantFunctionalArea, ApplicantEducation, ApplicantExperience, ApplicantAward, ApplicantCertificate
from db.database import SessionLocal, engine
from db.models import Base

fake = Faker()

# Define job titles by functional area
JOB_TITLES = {
    "Software Engineering": [
        "Software Engineer", "Frontend Developer", "Backend Developer", 
        "Full Stack Developer", "DevOps Engineer", "QA Engineer",
        "AI Specialist", "Data Engineer", "Machine Learning Engineer",
        "Mobile App Developer", "System Architect", "Tech Lead"
    ],
    "Data Science": [
        "Data Scientist", "Data Analyst", "Business Intelligence Analyst",
        "Machine Learning Engineer", "Data Engineer", "Statistician",
        "Research Scientist", "Data Architect", "BI Developer"
    ],
    "Marketing": [
        "Marketing Manager", "Digital Marketing Specialist", "SEO Specialist",
        "Content Marketing Manager", "Social Media Manager", "Marketing Analyst",
        "Brand Manager", "Growth Hacker", "Marketing Coordinator"
    ],
    "Sales": [
        "Sales Manager", "Sales Representative", "Account Executive",
        "Business Development Manager", "Sales Engineer", "Inside Sales Rep",
        "Outside Sales Rep", "Sales Consultant", "Channel Sales Manager"
    ],
    "Human Resources": [
        "HR Manager", "HR Specialist", "Recruiter", "Talent Acquisition",
        "HR Business Partner", "Compensation Analyst", "Training Specialist",
        "HR Coordinator", "People Operations Manager"
    ],
    "Finance": [
        "Financial Analyst", "Accountant", "Financial Manager", "Controller",
        "Investment Analyst", "Risk Manager", "Compliance Officer",
        "Tax Specialist", "Treasury Analyst", "CFO"
    ],
    "Operations": [
        "Operations Manager", "Project Manager", "Scrum Master", "Product Manager",
        "Operations Analyst", "Process Engineer", "Supply Chain Manager",
        "Logistics Coordinator", "Quality Assurance Manager"
    ],
    "Design": [
        "UI/UX Designer", "Graphic Designer", "Product Designer", "Art Director",
        "Visual Designer", "Interaction Designer", "Motion Designer",
        "Brand Designer", "Web Designer", "Creative Director"
    ]
}

# Define required skills by functional area
SKILLS_BY_AREA = {
    "Software Engineering": [
        "Python", "JavaScript", "Java", "C#", "C++", "React", "Angular", "Vue.js",
        "Node.js", "Docker", "Kubernetes", "AWS", "Azure", "GCP", "SQL", "NoSQL",
        "Git", "CI/CD", "Agile", "Scrum", "Machine Learning", "AI", "Go", "Rust",
        "TypeScript", "GraphQL", "REST APIs", "PostgreSQL", "MongoDB", "Redis"
    ],
    "Data Science": [
        "Python", "R", "SQL", "Pandas", "NumPy", "Scikit-learn", "TensorFlow",
        "PyTorch", "Tableau", "Power BI", "Excel", "Hadoop", "Spark", "Statistical Analysis",
        "Data Visualization", "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
        "Big Data", "Data Mining", "Predictive Modeling", "A/B Testing"
    ],
    "Marketing": [
        "Digital Marketing", "SEO", "Google Ads", "Facebook Ads", "Content Marketing",
        "Social Media Marketing", "Email Marketing", "Marketing Automation",
        "Google Analytics", "Marketing Strategy", "Brand Management", "Copywriting",
        "Influencer Marketing", "Conversion Rate Optimization", "CRM", "Marketing Research"
    ],
    "Sales": [
        "Sales", "Negotiation", "Relationship Building", "Lead Generation", "CRM",
        "Account Management", "Sales Presentations", "Consultative Selling",
        "Sales Forecasting", "Prospecting", "Closing", "B2B Sales", "Channel Sales",
        "Inside Sales", "Outside Sales", "Sales Strategy"
    ],
    "Human Resources": [
        "Recruitment", "Talent Acquisition", "HR Strategy", "Employee Relations",
        "Compensation & Benefits", "Training & Development", "Performance Management",
        "HRIS", "Employment Law", "Diversity & Inclusion", "Onboarding",
        "Succession Planning", "HR Analytics", "Conflict Resolution"
    ],
    "Finance": [
        "Financial Analysis", "Accounting", "Budgeting", "Forecasting", "Excel",
        "Financial Modeling", "Investment Analysis", "Risk Management", "Tax",
        "Audit", "Compliance", "Financial Planning", "FP&A", "M&A", "Treasury"
    ],
    "Operations": [
        "Project Management", "Process Improvement", "Lean Six Sigma", "Agile",
        "Scrum", "Operations Management", "Supply Chain", "Logistics",
        "Quality Assurance", "Change Management", "Stakeholder Management",
        "Resource Planning", "Workflow Optimization", "Continuous Improvement"
    ],
    "Design": [
        "UI/UX Design", "Figma", "Adobe Creative Suite", "Prototyping", "User Research",
        "Interaction Design", "Visual Design", "Wireframing", "Information Architecture",
        "User Testing", "Brand Design", "Graphic Design", "Motion Design",
        "Design Systems", "Typography", "Color Theory", "Illustration"
    ]
}

# Education levels
EDUCATION_LEVELS = ["high_school", "associate", "bachelor", "master", "phd"]

# Experience ranges by job level
EXPERIENCE_LEVELS = [
    {"min": 0, "max": 2, "level": "junior"},
    {"min": 2, "max": 5, "level": "mid"},
    {"min": 5, "max": 10, "level": "senior"},
    {"min": 10, "max": 20, "level": "lead"}
]

# Countries and currencies
COUNTRIES = [
    {"code": "US", "name": "United States", "currency": "USD"},
    {"code": "GB", "name": "United Kingdom", "currency": "GBP"},
    {"code": "DE", "name": "Germany", "currency": "EUR"},
    {"code": "FR", "name": "France", "currency": "EUR"},
    {"code": "CA", "name": "Canada", "currency": "CAD"},
    {"code": "AU", "name": "Australia", "currency": "AUD"},
    {"code": "IN", "name": "India", "currency": "INR"},
    {"code": "SG", "name": "Singapore", "currency": "SGD"},
    {"code": "JP", "name": "Japan", "currency": "JPY"},
    {"code": "AE", "name": "United Arab Emirates", "currency": "AED"}
]

def generate_job(area_name):
    """Generate a single job listing with realistic data."""
    title = random.choice(JOB_TITLES[area_name])
    experience_req = random.choice(EXPERIENCE_LEVELS)
    country = random.choice(COUNTRIES)
    
    # Generate salary based on experience level and area
    base_salary = {
        "Software Engineering": 80000,
        "Data Science": 90000,
        "Marketing": 50000,
        "Sales": 60000,
        "Human Resources": 55000,
        "Finance": 70000,
        "Operations": 65000,
        "Design": 60000
    }
    
    salary_multiplier = 1.0 + (experience_req["min"] * 0.15)
    min_salary = int(base_salary[area_name] * salary_multiplier * 0.7)
    max_salary = int(base_salary[area_name] * salary_multiplier * 1.3)
    
    # Generate job requirements based on required skills
    required_skills = random.sample(SKILLS_BY_AREA[area_name], random.randint(3, 8))
    requirements_html = "<ul>"
    for skill in required_skills:
        requirements_html += f"<li>{skill}</li>"
    requirements_html += "</ul>"
    
    # Generate role summary
    role_summaries = [
        f"<p>Join our dynamic team as a {title} to help drive innovation in {area_name}.</p>",
        f"<p>We're seeking a talented {title} to contribute to our growing {area_name} department.</p>",
        f"<p>Exciting opportunity for a {title} to make an impact in the {area_name} field.</p>",
        f"<p>Looking for a skilled {title} to join our {area_name} team and help us achieve our goals.</p>"
    ]
    
    return Job(
        title=title,
        reference=f"JOB{str(uuid.uuid4())[:8].upper()}",
        number_of_vacancies=random.randint(1, 3),
        status="published",
        is_open=random.choice([True, False]) if random.random() < 0.1 else True,  # 10% chance of closed
        job_requirements=requirements_html,
        role_summary=random.choice(role_summaries),
        duties_and_responsibilities="<ul><li>Perform assigned duties</li><li>Collaborate with team members</li><li>Meet project deadlines</li></ul>",
        minimum_experience=experience_req["min"],
        maximum_experience=experience_req["max"],
        education_level=random.choice(EDUCATION_LEVELS),
        salary_type="range",
        minimum_salary=min_salary,
        maximum_salary=max_salary,
        currency=country["currency"],
        period="yearly",
        work_type=random.choice(["full_time", "part_time", "contract", "remote"]),
        gender=random.choice(["any", "male", "female"]),
        nationality="any",
        language="any",
        publish_date=fake.date_between(start_date='-30d', end_date='today'),
        expiry_date=fake.date_between(start_date='today', end_date='+60d'),
        company_name=fake.company(),
        company_is_public=random.choice([True, False]),
        company_work_policy=random.choice(["remote", "on-site", "hybrid"]),
        industry_type_name=area_name,
        country_code=country["code"],
        country_name=country["name"],
        province_name=fake.state(),
        area_name=area_name
    )

def generate_applicant():
    """Generate a single applicant with realistic data."""
    # Choose random functional areas for the applicant
    num_areas = random.randint(1, 3)
    selected_areas = random.sample(list(SKILLS_BY_AREA.keys()), num_areas)
    
    # Generate skills from selected areas
    all_skills = []
    for area in selected_areas:
        area_skills = random.sample(SKILLS_BY_AREA[area], random.randint(3, 8))
        all_skills.extend(area_skills)
    # Remove duplicates and limit to reasonable number
    unique_skills = list(set(all_skills))[:15]
    
    # Generate experience
    num_experiences = random.randint(1, 5)
    experiences = []
    for _ in range(num_experiences):
        start_date = fake.date_between(start_date='-10y', end_date='-1y')
        end_date = fake.date_between(start_date=start_date, end_date='today') if random.choice([True, False]) else None
        experiences.append({
            'title': random.choice(JOB_TITLES[selected_areas[0]]),
            'company': fake.company(),
            'start_date': start_date,
            'end_date': end_date,
            'description': fake.text(max_nb_chars=200)
        })
    
    # Calculate years of experience based on experiences
    years_of_experience = 0
    for exp in experiences:
        if exp['end_date']:
            years_of_experience += (exp['end_date'] - exp['start_date']).days // 365
        else:
            years_of_experience += (datetime.today().date() - exp['start_date']).days // 365
    
    return {
        'applicant_data': Applicant(
            name=fake.name(),
            email=fake.email(),
            phone=fake.phone_number(),
            bio=fake.text(max_nb_chars=300),
            nationality=random.choice([country["name"] for country in COUNTRIES]),
            date_of_birth=fake.date_of_birth(minimum_age=22, maximum_age=65),
            gender=random.choice(["Male", "Female", "Other"]),
            address=fake.address(),
            country_code=random.choice(COUNTRIES)["code"],
            country_name=random.choice(COUNTRIES)["name"],
            province_name=fake.state(),
            speaking_languages='["English", "Spanish"]' if random.random() < 0.3 else '["English"]'
        ),
        'skills': unique_skills,
        'functional_areas': selected_areas,
        'education': {
            'level': random.choice(EDUCATION_LEVELS),
            'institute': fake.company(),  # Using company name as institute
            'field_of_study': random.choice(["Computer Science", "Business", "Engineering", "Mathematics", "Psychology", "Economics"])
        },
        'experiences': experiences,
        'awards': random.choice([[], [{"title": "Employee of the Month", "issuer": fake.company(), "date": fake.date_this_year()}]]) if random.random() < 0.3 else [],
        'certificates': random.choice([[], [{"title": "AWS Certified Solutions Architect", "issuer": "Amazon", "date": fake.date_this_decade()}]]) if random.random() < 0.4 else []
    }

def seed_database(db: Session, num_jobs: int = 1000, num_applicants: int = 20):
    """Seed the database with jobs and applicants."""
    print(f"Seeding database with {num_jobs} jobs and {num_applicants} applicants...")
    
    # Get all functional areas
    functional_areas = list(JOB_TITLES.keys())
    
    # Seed jobs
    jobs_created = 0
    for i in range(num_jobs):
        area = random.choice(functional_areas)
        job = generate_job(area)
        db.add(job)
        db.flush()  # Get the ID for relationships if needed
        jobs_created += 1
        
        if jobs_created % 100 == 0:
            print(f"Created {jobs_created}/{num_jobs} jobs...")
    
    db.commit()
    print(f"Successfully created {jobs_created} jobs.")
    
    # Seed applicants
    applicants_created = 0
    for i in range(num_applicants):
        applicant_data = generate_applicant()
        
        # Add applicant
        applicant = applicant_data['applicant_data']
        db.add(applicant)
        db.flush()  # Get the ID for relationships
        
        # Add skills
        for skill_name in applicant_data['skills']:
            skill = ApplicantSkill(
                applicant_id=applicant.id,
                skill_name=skill_name
            )
            db.add(skill)
        
        # Add functional areas
        for area_name in applicant_data['functional_areas']:
            area = ApplicantFunctionalArea(
                applicant_id=applicant.id,
                area_name=area_name
            )
            db.add(area)
        
        # Add education
        education = ApplicantEducation(
            applicant_id=applicant.id,
            level=applicant_data['education']['level'],
            institute_name=applicant_data['education']['institute'],
            field_of_study=applicant_data['education']['field_of_study']
        )
        db.add(education)
        
        # Add experiences
        for exp in applicant_data['experiences']:
            experience = ApplicantExperience(
                applicant_id=applicant.id,
                title=exp['title'],
                company_name=exp['company'],
                start_date=exp['start_date'],
                end_date=exp['end_date'],
                description=exp['description']
            )
            db.add(experience)
        
        # Add awards if any
        for award in applicant_data['awards']:
            award_obj = ApplicantAward(
                applicant_id=applicant.id,
                title=award['title'],
                issued_by=award['issuer'],
                issued_date=award['date']
            )
            db.add(award_obj)
        
        # Add certificates if any
        for cert in applicant_data['certificates']:
            cert_obj = ApplicantCertificate(
                applicant_id=applicant.id,
                title=cert['title'],
                issued_by=cert['issuer'],
                issued_date=cert['date']
            )
            db.add(cert_obj)
        
        applicants_created += 1
        
        if applicants_created % 5 == 0:
            print(f"Created {applicants_created}/{num_applicants} applicants...")
    
    db.commit()
    print(f"Successfully created {applicants_created} applicants.")
    print("Database seeding completed!")

def main():
    print("Initializing database tables...")
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully!")
    
    # Create database session
    db = SessionLocal()
    try:
        seed_database(db, num_jobs=1000, num_applicants=20)
    finally:
        db.close()

if __name__ == "__main__":
    main()