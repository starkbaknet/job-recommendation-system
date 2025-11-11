from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID
import json
from datetime import datetime

from db.database import get_db
from utils.vector_utils import get_recommendations_for_applicant, get_global_embedder, compute_job_vectors_batch

# Cold start: Load the model when the application starts
print("Loading model at startup...")
get_global_embedder()  # Initialize the model at startup
print("Model loaded successfully!")

app = FastAPI(
    title="Job Recommendation API",
    description="API for job recommendations based on applicant profiles with PostgreSQL backend",
    version="1.0.0"
)

class JobRecommendation(BaseModel):
    title: str
    reference: str
    location: str
    area_name: str
    minimum_salary: Optional[float]
    maximum_salary: Optional[float]
    salary_type: str
    gender: str
    period: str
    language: str
    publish_date: str
    closing_date: str
    similarity_score: float

class ApplicantResponse(BaseModel):
    id: str
    name: str
    email: str
    phone: Optional[str]
    bio: Optional[str]
    nationality: Optional[str]
    date_of_birth: Optional[str]
    gender: Optional[str]
    address: Optional[str]
    country_code: str
    country_name: str
    province_name: str
    speaking_languages: Optional[str]
    created_at: str
    updated_at: str
    skills: List[str]
    functional_areas: List[str]
    education: List[dict]
    experience: List[dict]

class PaginatedResponse(BaseModel):
    total: int
    page: int
    size: int
    total_pages: int
    data: List[JobRecommendation]

class PaginatedApplicantsResponse(BaseModel):
    total: int
    page: int
    size: int
    total_pages: int
    data: List[ApplicantResponse]

@app.get("/")
def read_root():
    return {"message": "Job Recommendation API - Connect to PostgreSQL database with UUID support"}

@app.get("/recommendations/{applicant_id}", response_model=List[JobRecommendation])
def get_job_recommendations(
    applicant_id: str,
    top_k: int = Query(10, ge=1, le=200, description="Number of recommendations to return"),
    min_similarity_threshold: float = Query(0.60, ge=0.0, le=1.0, description="Minimum similarity threshold (0.0-1.0). Default 0.60 for weighted scoring."),
    db: Session = Depends(get_db)
):
    """
    Get job recommendations for an applicant by their ID.
    Recommendations prioritize skills and experience over functional areas.
    Only jobs above the minimum similarity threshold are returned.
    """
    try:
        # Validate UUID format
        UUID(applicant_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid applicant ID format")
    
    try:
        recommendations = get_recommendations_for_applicant(db, applicant_id, top_k, min_similarity_threshold)
        return recommendations
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/recommendations/{applicant_id}/paginated", response_model=PaginatedResponse)
def get_paginated_job_recommendations(
    applicant_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    min_similarity_threshold: float = Query(0.60, ge=0.0, le=1.0, description="Minimum similarity threshold (0.0-1.0). Default 0.60 for weighted scoring."),
    db: Session = Depends(get_db)
):
    """
    Get paginated job recommendations for an applicant by their ID.
    Recommendations prioritize skills and experience over functional areas.
    Only jobs above the minimum similarity threshold are returned.
    """
    try:
        # Validate UUID format
        UUID(applicant_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid applicant ID format")
    
    try:
        # Get all recommendations for the applicant 
        # In a real implementation, this would be optimized with database-level pagination
        all_recommendations = get_recommendations_for_applicant(db, applicant_id, top_k=200, min_similarity_threshold=min_similarity_threshold)  # Get max possible
        
        total = len(all_recommendations)
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        paginated_data = all_recommendations[start_idx:end_idx]
        
        total_pages = (total + size - 1) // size  # Ceiling division
        
        return PaginatedResponse(
            total=total,
            page=page,
            size=size,
            total_pages=total_pages,
            data=paginated_data
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/jobs")
def get_jobs(
    area_name: Optional[str] = Query(None, description="Filter by functional area"),
    company_name: Optional[str] = Query(None, description="Filter by company name"),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    db: Session = Depends(get_db)
):
    """
    Get paginated list of jobs with optional filtering.
    """
    from db.models import Job
    
    query = db.query(Job)
    
    if area_name:
        query = query.filter(Job.area_name == area_name)
    if company_name:
        query = query.filter(Job.company_name == company_name)
    
    total = query.count()
    jobs = query.offset((page - 1) * size).limit(size).all()
    
    # Convert jobs to response format with all fields
    jobs_data = []
    for job in jobs:
        job_response = JobRecommendation(
            job_id=str(job.id),
            title=job.title,
            reference=job.reference,
            company=f"{job.company_name} ({'public' if job.company_is_public else 'private'})",
            company_name=job.company_name,
            company_is_public=job.company_is_public,
            company_work_policy=job.company_work_policy,
            location=f"{job.province_name}, {job.country_name}",
            country_code=job.country_code,
            country_name=job.country_name,
            province_name=job.province_name,
            area_name=job.area_name,
            industry_type_name=job.industry_type_name,
            number_of_vacancies=job.number_of_vacancies,
            status=job.status,
            is_open=job.is_open,
            minimum_experience=job.minimum_experience or 0,
            maximum_experience=job.maximum_experience or 20,
            education_level=job.education_level or "Not specified",
            salary_type=job.salary_type or "Not specified",
            minimum_salary=float(job.minimum_salary) if job.minimum_salary else None,
            maximum_salary=float(job.maximum_salary) if job.maximum_salary else None,
            currency=job.currency,
            period=job.period,
            work_type=job.work_type,
            gender=job.gender,
            nationality=job.nationality,
            language=job.language,
            publish_date=job.publish_date.isoformat() if job.publish_date else "Not specified",
            expiry_date=job.expiry_date.isoformat() if job.expiry_date else "Not specified",
            role_summary=job.role_summary or "",
            job_requirements=job.job_requirements or "",
            duties_and_responsibilities=job.duties_and_responsibilities or "",
            similarity_score=0.0  # Default for job listing
        )
        jobs_data.append(job_response)
    
    total_pages = (total + size - 1) // size
    
    return PaginatedResponse(
        total=total,
        page=page,
        size=size,
        total_pages=total_pages,
        data=jobs_data
    )

@app.get("/applicants", response_model=PaginatedApplicantsResponse)
def get_applicants(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    db: Session = Depends(get_db)
):
    """
    Get paginated list of applicants with all their details.
    """
    from db.models import Applicant, ApplicantSkill, ApplicantFunctionalArea, ApplicantEducation, ApplicantExperience
    
    total = db.query(Applicant).count()
    applicants = db.query(Applicant).offset((page - 1) * size).limit(size).all()
    
    # Convert applicants to response format with all fields
    applicants_data = []
    for applicant in applicants:
        # Fetch related data for full response
        skills = [skill.skill_name for skill in db.query(ApplicantSkill).filter(ApplicantSkill.applicant_id == applicant.id).all()]
        functional_areas = [area.area_name for area in db.query(ApplicantFunctionalArea).filter(ApplicantFunctionalArea.applicant_id == applicant.id).all()]
        education = [{"level": edu.level, "institute": edu.institute_name, "field_of_study": edu.field_of_study} 
                     for edu in db.query(ApplicantEducation).filter(ApplicantEducation.applicant_id == applicant.id).all()]
        experience = [{"title": exp.title, "company": exp.company_name, "start_date": exp.start_date.isoformat() if exp.start_date else None, 
                       "end_date": exp.end_date.isoformat() if exp.end_date else None, "description": exp.description} 
                      for exp in db.query(ApplicantExperience).filter(ApplicantExperience.applicant_id == applicant.id).all()]
        
        applicant_response = ApplicantResponse(
            id=str(applicant.id),
            name=applicant.name,
            email=applicant.email,
            phone=applicant.phone,
            bio=applicant.bio,
            nationality=applicant.nationality,
            date_of_birth=applicant.date_of_birth.isoformat() if applicant.date_of_birth else None,
            gender=applicant.gender,
            address=applicant.address,
            country_code=applicant.country_code,
            country_name=applicant.country_name,
            province_name=applicant.province_name,
            speaking_languages=applicant.speaking_languages,
            created_at=applicant.created_at.isoformat() if applicant.created_at else None,
            updated_at=applicant.updated_at.isoformat() if applicant.updated_at else None,
            skills=skills,
            functional_areas=functional_areas,
            education=education,
            experience=experience
        )
        applicants_data.append(applicant_response)
    
    total_pages = (total + size - 1) // size
    
    return PaginatedApplicantsResponse(
        total=total,
        page=page,
        size=size,
        total_pages=total_pages,
        data=applicants_data
    )

@app.get("/jobs/{job_id}", response_model=JobRecommendation)
def get_job_details(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Get full details of a job by its ID.
    """
    try:
        # Validate UUID format
        UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID format")
    
    from db.models import Job
    
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_response = JobRecommendation(
        job_id=str(job.id),
        title=job.title,
        reference=job.reference,
        company=f"{job.company_name} ({'public' if job.company_is_public else 'private'})",
        company_name=job.company_name,
        company_is_public=job.company_is_public,
        company_work_policy=job.company_work_policy,
        location=f"{job.province_name}, {job.country_name}",
        country_code=job.country_code,
        country_name=job.country_name,
        province_name=job.province_name,
        area_name=job.area_name,
        industry_type_name=job.industry_type_name,
        number_of_vacancies=job.number_of_vacancies,
        status=job.status,
        is_open=job.is_open,
        minimum_experience=job.minimum_experience or 0,
        maximum_experience=job.maximum_experience or 20,
        education_level=job.education_level or "Not specified",
        salary_type=job.salary_type or "Not specified",
        minimum_salary=float(job.minimum_salary) if job.minimum_salary else None,
        maximum_salary=float(job.maximum_salary) if job.maximum_salary else None,
        currency=job.currency,
        period=job.period,
        work_type=job.work_type,
        gender=job.gender,
        nationality=job.nationality,
        language=job.language,
        publish_date=job.publish_date.isoformat() if job.publish_date else "Not specified",
        expiry_date=job.expiry_date.isoformat() if job.expiry_date else "Not specified",
        role_summary=job.role_summary or "",
        job_requirements=job.job_requirements or "",
        duties_and_responsibilities=job.duties_and_responsibilities or "",
        similarity_score=0.0  # Default for individual job lookup
    )
    
    return job_response

@app.get("/applicants/{applicant_id}", response_model=ApplicantResponse)
def get_applicant_details(
    applicant_id: str,
    db: Session = Depends(get_db)
):
    """
    Get full details of an applicant by their ID.
    """
    try:
        # Validate UUID format
        UUID(applicant_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid applicant ID format")
    
    from db.models import Applicant, ApplicantSkill, ApplicantFunctionalArea, ApplicantEducation, ApplicantExperience
    
    applicant = db.query(Applicant).filter(Applicant.id == applicant_id).first()
    if not applicant:
        raise HTTPException(status_code=404, detail="Applicant not found")
    
    # Fetch related data for full response
    skills = [skill.skill_name for skill in db.query(ApplicantSkill).filter(ApplicantSkill.applicant_id == applicant.id).all()]
    functional_areas = [area.area_name for area in db.query(ApplicantFunctionalArea).filter(ApplicantFunctionalArea.applicant_id == applicant.id).all()]
    education = [{"level": edu.level, "institute": edu.institute_name, "field_of_study": edu.field_of_study} 
                 for edu in db.query(ApplicantEducation).filter(ApplicantEducation.applicant_id == applicant.id).all()]
    experience = [{"title": exp.title, "company": exp.company_name, "start_date": exp.start_date.isoformat() if exp.start_date else None, 
                   "end_date": exp.end_date.isoformat() if exp.end_date else None, "description": exp.description} 
                  for exp in db.query(ApplicantExperience).filter(ApplicantExperience.applicant_id == applicant.id).all()]
    
    applicant_response = ApplicantResponse(
        id=str(applicant.id),
        name=applicant.name,
        email=applicant.email,
        phone=applicant.phone,
        bio=applicant.bio,
        nationality=applicant.nationality,
        date_of_birth=applicant.date_of_birth.isoformat() if applicant.date_of_birth else None,
        gender=applicant.gender,
        address=applicant.address,
        country_code=applicant.country_code,
        country_name=applicant.country_name,
        province_name=applicant.province_name,
        speaking_languages=applicant.speaking_languages,
        created_at=applicant.created_at.isoformat() if applicant.created_at else None,
        updated_at=applicant.updated_at.isoformat() if applicant.updated_at else None,
        skills=skills,
        functional_areas=functional_areas,
        education=education,
        experience=experience
    )
    
    return applicant_response

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# Manual trigger for vectorization
@app.post("/trigger-vectorization")
def trigger_vectorization(batch_size: int = Query(50, ge=1, le=1000, description="Batch size for processing")):
    """
    Manually trigger vectorization of jobs that don't have vectors yet.
    """
    try:
        db = next(get_db())
        processed_count = compute_job_vectors_batch(db, batch_size)
        db.close()
        return {"message": f"Vectorization completed", "processed_count": processed_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during vectorization: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)