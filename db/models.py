from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, Date, DateTime, DECIMAL, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func
import uuid
from typing import List, Optional

Base = declarative_base()

class Job(Base):
    __tablename__ = 'jobs'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    reference = Column(String(50), unique=True, nullable=False)
    number_of_vacancies = Column(Integer, default=1)
    status = Column(String(20), default='draft')
    is_open = Column(Boolean, default=True)
    job_requirements = Column(Text)
    role_summary = Column(Text)
    duties_and_responsibilities = Column(Text)
    minimum_experience = Column(Integer)
    maximum_experience = Column(Integer)
    education_level = Column(String(50))
    salary_type = Column(String(20))
    minimum_salary = Column(DECIMAL(10, 2))
    maximum_salary = Column(DECIMAL(10, 2))
    currency = Column(String(10), default='USD')
    period = Column(String(20), default='monthly')
    work_type = Column(String(20), default='full_time')
    gender = Column(String(20), default='any')
    nationality = Column(String(100))
    language = Column(String(100))
    publish_date = Column(Date)
    expiry_date = Column(Date)
    company_id = Column(UUID(as_uuid=True))
    company_name = Column(String(255))
    company_is_public = Column(Boolean, default=False)
    company_work_policy = Column(String(20), default='hybrid')  # remote, on-site, hybrid
    industry_type_id = Column(String(50))
    industry_type_name = Column(String(100))
    country_code = Column(String(10))
    country_name = Column(String(100))
    province_name = Column(String(100))
    area_name = Column(String(100))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationship
    vector = relationship("JobVector", uselist=False, back_populates="job", cascade="all, delete-orphan")


class Applicant(Base):
    __tablename__ = 'applicants'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    phone = Column(String(50))
    bio = Column(Text)
    nationality = Column(String(100))
    date_of_birth = Column(Date)
    gender = Column(String(20))
    address = Column(Text)
    country_code = Column(String(10))
    country_name = Column(String(100))
    province_name = Column(String(100))
    speaking_languages = Column(Text)  # JSON array of languages as text
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    skills = relationship("ApplicantSkill", back_populates="applicant", cascade="all, delete-orphan")
    functional_areas = relationship("ApplicantFunctionalArea", back_populates="applicant", cascade="all, delete-orphan")
    education = relationship("ApplicantEducation", back_populates="applicant", cascade="all, delete-orphan")
    experience = relationship("ApplicantExperience", back_populates="applicant", cascade="all, delete-orphan")
    awards = relationship("ApplicantAward", back_populates="applicant", cascade="all, delete-orphan")
    certificates = relationship("ApplicantCertificate", back_populates="applicant", cascade="all, delete-orphan")
    vector = relationship("ApplicantVector", uselist=False, back_populates="applicant", cascade="all, delete-orphan")
    recommendations = relationship("JobRecommendationsCache", back_populates="applicant")


class ApplicantSkill(Base):
    __tablename__ = 'applicant_skills'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    applicant_id = Column(UUID(as_uuid=True), ForeignKey('applicants.id', ondelete='CASCADE'), nullable=False)
    skill_name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=func.now())

    # Relationship
    applicant = relationship("Applicant", back_populates="skills")


class ApplicantFunctionalArea(Base):
    __tablename__ = 'applicant_functional_areas'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    applicant_id = Column(UUID(as_uuid=True), ForeignKey('applicants.id', ondelete='CASCADE'), nullable=False)
    area_name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=func.now())

    # Relationship
    applicant = relationship("Applicant", back_populates="functional_areas")


class ApplicantEducation(Base):
    __tablename__ = 'applicant_education'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    applicant_id = Column(UUID(as_uuid=True), ForeignKey('applicants.id', ondelete='CASCADE'), nullable=False)
    level = Column(String(50), nullable=False)  # high_school, associate, bachelor, master, phd
    institute_name = Column(String(255), nullable=False)
    field_of_study = Column(String(255))
    start_date = Column(Date)
    end_date = Column(Date)
    created_at = Column(DateTime, default=func.now())

    # Relationship
    applicant = relationship("Applicant", back_populates="education")


class ApplicantExperience(Base):
    __tablename__ = 'applicant_experience'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    applicant_id = Column(UUID(as_uuid=True), ForeignKey('applicants.id', ondelete='CASCADE'), nullable=False)
    title = Column(String(255), nullable=False)
    company_name = Column(String(255), nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date)
    description = Column(Text)
    created_at = Column(DateTime, default=func.now())

    # Relationship
    applicant = relationship("Applicant", back_populates="experience")


class ApplicantAward(Base):
    __tablename__ = 'applicant_awards'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    applicant_id = Column(UUID(as_uuid=True), ForeignKey('applicants.id', ondelete='CASCADE'), nullable=False)
    title = Column(String(255), nullable=False)
    issued_by = Column(String(255), nullable=False)
    issued_date = Column(Date)
    created_at = Column(DateTime, default=func.now())

    # Relationship
    applicant = relationship("Applicant", back_populates="awards")


class ApplicantCertificate(Base):
    __tablename__ = 'applicant_certificates'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    applicant_id = Column(UUID(as_uuid=True), ForeignKey('applicants.id', ondelete='CASCADE'), nullable=False)
    title = Column(String(255), nullable=False)
    issued_by = Column(String(255), nullable=False)
    issued_date = Column(Date)
    created_at = Column(DateTime, default=func.now())

    # Relationship
    applicant = relationship("Applicant", back_populates="certificates")


class JobVector(Base):
    __tablename__ = 'job_vectors'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey('jobs.id', ondelete='CASCADE'), unique=True, nullable=False)
    # For now, storing as JSON text; in a real implementation with pgvector,
    # we would use a vector column type
    embedding = Column(Text)  # Will store as JSON array of floats
    computed_at = Column(DateTime, default=func.now())

    # Relationship
    job = relationship("Job", back_populates="vector")


class ApplicantVector(Base):
    __tablename__ = 'applicant_vectors'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    applicant_id = Column(UUID(as_uuid=True), ForeignKey('applicants.id', ondelete='CASCADE'), unique=True, nullable=False)
    # For now, storing as JSON text; in a real implementation with pgvector,
    # we would use a vector column type
    embedding = Column(Text)  # Will store as JSON array of floats
    computed_at = Column(DateTime, default=func.now())

    # Relationship
    applicant = relationship("Applicant", back_populates="vector")


class JobRecommendationsCache(Base):
    __tablename__ = 'job_recommendations_cache'
    __table_args__ = (UniqueConstraint('applicant_id', 'job_id', name='unique_applicant_job'),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    applicant_id = Column(UUID(as_uuid=True), ForeignKey('applicants.id', ondelete='CASCADE'), nullable=False)
    job_id = Column(UUID(as_uuid=True), ForeignKey('jobs.id', ondelete='CASCADE'), nullable=False)
    similarity_score = Column(DECIMAL(5, 4))
    computed_at = Column(DateTime, default=func.now())

    # Relationships
    applicant = relationship("Applicant", back_populates="recommendations")
    job = relationship("Job")