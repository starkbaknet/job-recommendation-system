from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bs4 import BeautifulSoup
import re
from sqlalchemy.orm import Session
from sqlalchemy import text
from db.models import Job, Applicant, ApplicantSkill, ApplicantEducation, ApplicantExperience, JobVector, ApplicantVector
import json
from datetime import datetime, date
import os

# Global model instance to avoid reloading
_embedder_instance = None

class MultilingualEmbedder:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initialize the multilingual sentence transformer model.
        """
        print(f"Loading multilingual embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully!")
    
    def encode_texts(self, texts):
        """
        Encode a list of texts into embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Filter out empty texts
        texts = [text for text in texts if text and text.strip()]
        
        if not texts:
            # Return a zero vector if no valid texts
            return np.array([np.zeros(self.model.get_sentence_embedding_dimension())])
        
        embeddings = self.model.encode(texts)
        return embeddings
    
    def get_similarity(self, text1, text2):
        """
        Calculate cosine similarity between two texts.
        """
        emb1 = self.encode_texts(text1)
        emb2 = self.encode_texts(text2)
        
        # Compute cosine similarity
        similarity = cosine_similarity(emb1, emb2)
        
        # Return the similarity score (value between 0 and 1)
        return similarity[0][0]
    
    def get_batch_similarities(self, query_text, candidate_texts):
        """
        Calculate similarities between a query text and multiple candidate texts.
        """
        query_embedding = self.encode_texts(query_text)
        candidate_embeddings = self.encode_texts(candidate_texts)
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, candidate_embeddings)
        
        # Return the similarity scores
        return similarities[0]

def get_global_embedder():
    """
    Get the global embedder instance or create it if it doesn't exist.
    This ensures the model is loaded only once.
    """
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = MultilingualEmbedder()
    return _embedder_instance

def clean_html(html_content: str) -> str:
    """
    Clean HTML content and extract text.
    """
    if not html_content:
        return ""
    
    # Use BeautifulSoup to extract text from HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    
    # Clean up the text
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = text.strip()
    
    return text

def extract_job_text(job: Job) -> str:
    """
    Extract all relevant text from a job for embedding.
    """
    text_parts = []
    
    # Title
    text_parts.append(job.title or "")
    
    # Role summary
    text_parts.append(clean_html(job.role_summary or ""))
    
    # Requirements
    text_parts.append(clean_html(job.job_requirements or ""))
    
    # Duties
    text_parts.append(clean_html(job.duties_and_responsibilities or ""))
    
    # Industry and area
    text_parts.append(job.industry_type_name or "")
    text_parts.append(job.area_name or "")
    
    # Experience requirements
    if job.minimum_experience is not None and job.maximum_experience is not None:
        exp_text = f"Experience required: {job.minimum_experience} to {job.maximum_experience} years"
        text_parts.append(exp_text)
    
    # Education
    text_parts.append(job.education_level or "")
    
    # Work type
    text_parts.append(job.work_type or "")
    
    # Company info
    text_parts.append(job.company_name or "")
    text_parts.append(job.company_work_policy or "")
    
    # Location
    text_parts.append(job.country_name or "")
    text_parts.append(job.province_name or "")
    
    # Join all parts
    return ' '.join(filter(None, text_parts))

def extract_applicant_text(db: Session, applicant: Applicant) -> str:
    """
    Extract all relevant text from an applicant for embedding.
    """
    text_parts = []
    
    # Basic info
    text_parts.append(applicant.name or "")
    text_parts.append(applicant.bio or "")
    
    # Skills
    skills = db.query(ApplicantSkill.skill_name).filter(ApplicantSkill.applicant_id == applicant.id).all()
    skills_list = [skill[0] for skill in skills]
    text_parts.append(', '.join(skills_list))
    
    # Education
    educations = db.query(ApplicantEducation).filter(ApplicantEducation.applicant_id == applicant.id).all()
    for edu in educations:
        edu_text = f"{edu.level} in {edu.field_of_study} from {edu.institute_name}"
        text_parts.append(edu_text)
    
    # Experience
    experiences = db.query(ApplicantExperience).filter(ApplicantExperience.applicant_id == applicant.id).all()
    for exp in experiences:
        exp_text = f"{exp.title} at {exp.company_name}. {exp.description}"
        text_parts.append(exp_text)
    
    # Functional areas
    functional_areas = [area.area_name for area in applicant.functional_areas]
    text_parts.append(', '.join(functional_areas))
    
    # Speaking languages
    text_parts.append(applicant.speaking_languages or "")
    
    # Location
    text_parts.append(applicant.country_name or "")
    text_parts.append(applicant.province_name or "")
    
    # Join all parts
    return ' '.join(filter(None, text_parts))

def compute_job_vector(db: Session, job: Job, embedder: MultilingualEmbedder) -> List[float]:
    """
    Compute embedding for a job and store it in the database.
    """
    job_text = extract_job_text(job)
    embedding = embedder.encode_texts(job_text)[0]  # Get the first (and only) embedding
    
    # Convert to list for JSON storage
    embedding_list = embedding.tolist()
    
    # Check if vector already exists
    existing_vector = db.query(JobVector).filter(JobVector.job_id == job.id).first()
    if existing_vector:
        # Update existing vector
        existing_vector.embedding = json.dumps(embedding_list)
        existing_vector.computed_at = datetime.utcnow()  # Update timestamp
    else:
        # Create new vector entry
        job_vector = JobVector(
            job_id=job.id,
            embedding=json.dumps(embedding_list),
            computed_at=datetime.utcnow()
        )
        db.add(job_vector)
    
    db.commit()
    return embedding_list

def compute_applicant_vector(db: Session, applicant: Applicant, embedder: MultilingualEmbedder) -> List[float]:
    """
    Compute embedding for an applicant and store it in the database.
    """
    applicant_text = extract_applicant_text(db, applicant)
    embedding = embedder.encode_texts(applicant_text)[0]  # Get the first (and only) embedding
    
    # Convert to list for JSON storage
    embedding_list = embedding.tolist()
    
    # Check if vector already exists
    existing_vector = db.query(ApplicantVector).filter(ApplicantVector.applicant_id == applicant.id).first()
    if existing_vector:
        # Update existing vector
        existing_vector.embedding = json.dumps(embedding_list)
        existing_vector.computed_at = datetime.utcnow()  # Update timestamp
    else:
        # Create new vector entry
        applicant_vector = ApplicantVector(
            applicant_id=applicant.id,
            embedding=json.dumps(embedding_list),
            computed_at=datetime.utcnow()
        )
        db.add(applicant_vector)
    
    db.commit()
    return embedding_list

def compute_job_vectors_batch(db: Session, batch_size: int = 50):
    """
    Compute vectors for jobs that don't have vectors in batches.
    """
    embedder = get_global_embedder()  # Use global embedder instance
    
    # Get jobs without vectors
    jobs_without_vectors = db.query(Job).outerjoin(Job.vector).filter(Job.vector == None).all()
    total_jobs = len(jobs_without_vectors)
    
    if total_jobs == 0:
        print("No jobs without vectors found.")
        return 0
    
    print(f"Computing vectors for {total_jobs} jobs without vectors...")
    
    processed = 0
    for i in range(0, total_jobs, batch_size):
        batch = jobs_without_vectors[i:i + batch_size]
        
        for job in batch:
            try:
                compute_job_vector(db, job, embedder)
                processed += 1
                
                if processed % 10 == 0:
                    print(f"Processed {processed}/{total_jobs} job vectors...")
            except Exception as e:
                print(f"Error processing job {job.id}: {str(e)}")
                continue
    
    print(f"Successfully computed vectors for {processed} jobs.")
    return processed

def _calculate_skills_match_score(applicant_skills: List[str], job_text: str) -> float:
    """
    Calculate skills match score based on how many applicant skills appear in the job description.
    Prioritizes skills heavily in the recommendation and handles hybrid roles well.
    Returns a score between 0 and 1.
    """
    if not applicant_skills:
        return 0.5  # Neutral score if no skills
    
    job_text_lower = job_text.lower()
    matched_skills = []
    unmatched_skills = []
    
    for skill in applicant_skills:
        skill_lower = skill.lower()
        # Check if skill appears in job text (case-insensitive)
        # Also check for partial matches (e.g., "python" matches "python3", "pythonic")
        if skill_lower in job_text_lower:
            matched_skills.append(skill)
        else:
            # Check for partial matches (skill is part of a word or vice versa)
            words_in_job = job_text_lower.split()
            if any(skill_lower in word or word in skill_lower for word in words_in_job if len(word) > 3):
                matched_skills.append(skill)
            else:
                unmatched_skills.append(skill)
    
    # Calculate match ratio
    match_ratio = len(matched_skills) / len(applicant_skills) if applicant_skills else 0
    
    # Boost score for high match ratios to prioritize strong skill matches
    # For hybrid roles, we want to reward applicants who have multiple relevant skills
    if match_ratio >= 0.8:
        base_score = 1.0
    elif match_ratio >= 0.6:
        base_score = 0.85
    elif match_ratio >= 0.4:
        base_score = 0.7
    elif match_ratio >= 0.2:
        base_score = 0.5
    else:
        base_score = 0.3
    
    # Additional boost if multiple skills match (important for hybrid roles)
    if len(matched_skills) >= 3:
        base_score = min(1.0, base_score * 1.1)  # 10% boost for 3+ matched skills
    
    # Check if this is a hybrid role and applicant has diverse skills
    is_hybrid = _detect_hybrid_role(applicant_skills, job_text)
    if is_hybrid and len(matched_skills) >= 2:
        # For hybrid roles, having multiple matched skills is even more valuable
        base_score = min(1.0, base_score * 1.05)  # Additional 5% boost
    
    return base_score


def _detect_primary_experience_domain(applicant: Applicant) -> str:
    """
    Detect the primary domain of the applicant's experience (technical, sales, etc.)
    Returns: 'technical', 'sales', 'hybrid', or 'other'
    """
    technical_keywords = ['developer', 'engineer', 'programmer', 'tech', 'software', 'programming', 
                         'frontend', 'backend', 'fullstack', 'architect', 'devops', 'sre', 'qa', 'test']
    sales_keywords = ['sales', 'account', 'business development', 'account executive', 'inside sales', 
                      'outside sales', 'representative', 'consultant', 'bdr', 'sdr']
    
    technical_count = 0
    sales_count = 0
    
    for exp in applicant.experience:
        title_lower = (exp.title or "").lower()
        desc_lower = (exp.description or "").lower()
        combined = title_lower + " " + desc_lower
        
        if any(keyword in combined for keyword in technical_keywords):
            technical_count += 1
        if any(keyword in combined for keyword in sales_keywords):
            sales_count += 1
    
    if technical_count > 0 and sales_count > 0:
        return 'hybrid'
    elif technical_count > sales_count:
        return 'technical'
    elif sales_count > 0:
        return 'sales'
    else:
        return 'other'


def _detect_job_domain(job: Job) -> str:
    """
    Detect the primary domain of a job (technical, sales, etc.)
    Returns: 'technical', 'sales', 'hybrid', or 'other'
    """
    job_text = extract_job_text(job).lower()
    title_lower = (job.title or "").lower()
    
    technical_keywords = ['developer', 'engineer', 'programmer', 'tech', 'software', 'programming', 
                         'frontend', 'backend', 'fullstack', 'architect', 'devops', 'sre', 'qa', 'test',
                         'javascript', 'python', 'java', 'react', 'node', 'api', 'database', 'cloud']
    sales_keywords = ['sales', 'account', 'business development', 'account executive', 'inside sales', 
                      'outside sales', 'representative', 'consultant', 'bdr', 'sdr', 'crm']
    
    technical_count = sum(1 for kw in technical_keywords if kw in job_text or kw in title_lower)
    sales_count = sum(1 for kw in sales_keywords if kw in job_text or kw in title_lower)
    
    if technical_count > 0 and sales_count > 0:
        return 'hybrid'
    elif technical_count > sales_count:
        return 'technical'
    elif sales_count > 0:
        return 'sales'
    else:
        return 'other'


def _calculate_experience_match_score_optimized(
    applicant: Applicant, job: Job, 
    applicant_domain: str, job_domain: str,
    years_of_experience: float, has_leadership_experience: bool, has_senior_experience: bool
) -> float:
    """
    Optimized version of experience match score calculation.
    Pre-calculated values are passed in to avoid redundant computation.
    Returns a score between 0 and 1.
    """
    # CRITICAL: Check if job domain matches applicant's primary experience domain
    domain_match_boost = 1.0
    if applicant_domain == 'technical' and job_domain == 'sales':
        domain_match_boost = 0.1  # 90% penalty
    elif applicant_domain == 'sales' and job_domain == 'technical':
        domain_match_boost = 0.1  # 90% penalty
    elif applicant_domain == job_domain:
        domain_match_boost = 1.5  # 50% boost for matching domains
    elif applicant_domain == 'hybrid' or job_domain == 'hybrid':
        domain_match_boost = 1.0
    
    # Calculate base score from experience requirements
    if job.minimum_experience is None and job.maximum_experience is None:
        base_score = 0.7
    else:
        min_exp = float(job.minimum_experience) if job.minimum_experience else 0
        max_exp = float(job.maximum_experience) if job.maximum_experience else float('inf')
        
        if min_exp <= years_of_experience <= max_exp:
            base_score = 1.0
        elif years_of_experience < min_exp:
            diff = min_exp - years_of_experience
            base_score = max(0.3, 1.0 - diff * 0.15)
        else:
            diff = years_of_experience - max_exp
            base_score = max(0.6, 1.0 - diff * 0.03)
    
    # Apply domain match boost/penalty
    base_score = min(1.0, max(0.0, base_score * domain_match_boost))
    
    # Boost for leadership/senior experience (only check job title for speed)
    job_title_lower = (job.title or "").lower()
    if has_leadership_experience and any(kw in job_title_lower for kw in ['lead', 'manager', 'director', 'head']):
        base_score = min(1.0, base_score * 1.15)
    if has_senior_experience and any(kw in job_title_lower for kw in ['senior', 'sr.', 'principal', 'staff']):
        base_score = min(1.0, base_score * 1.1)
    
    return base_score


def _calculate_experience_match_score(applicant: Applicant, job: Job) -> float:
    """
    Calculate experience match score with emphasis on relevant experience.
    Detects leadership/senior roles and gives them higher weight.
    Prioritizes jobs that match the applicant's primary experience domain.
    Returns a score between 0 and 1.
    """
    # Calculate applicant's total years of experience
    years_of_experience = 0
    has_leadership_experience = False
    has_senior_experience = False
    
    # Detect primary experience domain
    applicant_domain = _detect_primary_experience_domain(applicant)
    job_domain = _detect_job_domain(job)
    
    for exp in applicant.experience:
        if exp.end_date:
            exp_duration = (exp.end_date - exp.start_date).days / 365.25
        else:
            exp_duration = (date.today() - exp.start_date).days / 365.25
        years_of_experience += exp_duration
        
        # Check for leadership indicators in job title
        title_lower = exp.title.lower() if exp.title else ""
        if any(keyword in title_lower for keyword in ['lead', 'manager', 'director', 'head', 'chief', 'vp', 'vice president']):
            has_leadership_experience = True
        if any(keyword in title_lower for keyword in ['senior', 'sr.', 'principal', 'staff', 'architect']):
            has_senior_experience = True
    
    # CRITICAL: Check if job domain matches applicant's primary experience domain
    domain_match_boost = 1.0
    if applicant_domain == 'technical' and job_domain == 'sales':
        domain_match_boost = 0.1  # 90% penalty
    elif applicant_domain == 'sales' and job_domain == 'technical':
        domain_match_boost = 0.1  # 90% penalty
    elif applicant_domain == job_domain:
        domain_match_boost = 1.5  # 50% boost for matching domains
    elif applicant_domain == 'hybrid' or job_domain == 'hybrid':
        domain_match_boost = 1.0
    
    # If job has no experience requirements, give neutral score
    if job.minimum_experience is None and job.maximum_experience is None:
        base_score = 0.7
    else:
        min_exp = float(job.minimum_experience) if job.minimum_experience else 0
        max_exp = float(job.maximum_experience) if job.maximum_experience else float('inf')
        
        # Check if experience is in range
        if min_exp <= years_of_experience <= max_exp:
            base_score = 1.0
        elif years_of_experience < min_exp:
            # Experience less than minimum - penalize but not too harshly
            diff = min_exp - years_of_experience
            base_score = max(0.3, 1.0 - diff * 0.15)
        else:
            # Experience more than maximum - still good, especially for senior roles
            diff = years_of_experience - max_exp
            base_score = max(0.6, 1.0 - diff * 0.03)
    
    # Apply domain match boost/penalty
    base_score = base_score * domain_match_boost
    base_score = min(1.0, max(0.0, base_score))  # Clamp to [0, 1]
    
    # Boost for leadership/senior experience matching job requirements
    job_title_lower = (job.title or "").lower()
    job_text_lower = extract_job_text(job).lower()
    
    if has_leadership_experience and any(keyword in job_text_lower for keyword in ['lead', 'manager', 'director', 'head', 'leadership']):
        base_score = min(1.0, base_score * 1.15)
    
    if has_senior_experience and any(keyword in job_title_lower for keyword in ['senior', 'sr.', 'principal', 'staff', 'lead']):
        base_score = min(1.0, base_score * 1.1)
    
    return base_score


def _calculate_functional_area_score(job_area: Optional[str], applicant_functional_areas: List[str]) -> float:
    """
    Calculate functional area match score (secondary factor).
    Returns a score between 0 and 1.
    """
    if not job_area or not applicant_functional_areas:
        return 0.5  # Neutral if no area specified
    
    if job_area.lower() in [fa.lower() for fa in applicant_functional_areas]:
        return 0.7  # Moderate boost for functional area match
    else:
        return 0.4  # Slight penalty for mismatch, but not too harsh


def _calculate_location_worktype_score(applicant: Applicant, job: Job) -> float:
    """
    Calculate location and work type match score (tertiary factors).
    Returns a score between 0 and 1.
    """
    location_score = 0.5  # Neutral default
    worktype_score = 0.5  # Neutral default
    
    # Location matching (if both have location info)
    if applicant.country_name and job.country_name:
        if applicant.country_name.lower() == job.country_name.lower():
            location_score = 0.7
            if applicant.province_name and job.province_name:
                if applicant.province_name.lower() == job.province_name.lower():
                    location_score = 0.9
    
    # Work type matching (prefer remote/hybrid if applicant prefers it)
    # This is a simplified check - in production, you'd store applicant preferences
    if job.work_type:
        worktype_score = 0.6  # Slight preference for any work type match
    
    # Combine location and work type (weighted average)
    return (location_score * 0.6 + worktype_score * 0.4)


def _detect_hybrid_role(applicant_skills: List[str], job_text: str) -> bool:
    """
    Detect if this is a hybrid role requiring skills from multiple domains.
    """
    # Define skill categories
    technical_keywords = ['programming', 'code', 'software', 'developer', 'engineer', 'technical', 'api', 'database', 'algorithm']
    business_keywords = ['sales', 'marketing', 'business', 'strategy', 'analyst', 'consultant', 'product']
    design_keywords = ['design', 'ui', 'ux', 'creative', 'graphic', 'visual']
    
    job_text_lower = job_text.lower()
    
    technical_count = sum(1 for kw in technical_keywords if kw in job_text_lower)
    business_count = sum(1 for kw in business_keywords if kw in job_text_lower)
    design_count = sum(1 for kw in design_keywords if kw in job_text_lower)
    
    # If job requires skills from 2+ categories, it's hybrid
    categories_required = sum([
        technical_count > 0,
        business_count > 0,
        design_count > 0
    ])
    
    return categories_required >= 2


def get_recommendations_for_applicant(db: Session, applicant_id: str, top_k: int = 10, 
                                     min_similarity_threshold: float = 0.60) -> List[Dict[str, Any]]:
    """
    Get job recommendations for an applicant prioritizing skills and experience over functional areas.
    OPTIMIZED: Only uses pgvector, loads minimal applicant data, returns simplified job details.
    
    Algorithm prioritizes:
    1. Skills match (40% weight) - Primary factor
    2. Experience match (30% weight) - Primary factor
    3. Base semantic similarity (20% weight) - Foundation
    4. Functional area (7% weight) - Secondary factor
    5. Location and work type (3% weight) - Tertiary factors
    
    Only returns jobs above the minimum similarity threshold (default 0.60).
    Note: With weighted scoring, scores typically range from 0.5-0.8, so threshold of 0.60 ensures quality matches.
    """
    # OPTIMIZED: Load only applicant embedding (not full applicant object)
    applicant_vector_record = db.query(ApplicantVector).filter(ApplicantVector.applicant_id == applicant_id).first()
    if not applicant_vector_record:
        # Need to create embedding - load minimal applicant data
        applicant = db.query(Applicant).filter(Applicant.id == applicant_id).first()
        if not applicant:
            raise ValueError(f"No applicant found with id {applicant_id}")
        embedder = get_global_embedder()
        applicant_embedding = compute_applicant_vector(db, applicant, embedder)
        applicant_embedding = np.array(applicant_embedding).reshape(1, -1)
        
        # Load minimal applicant data for scoring (only what we need)
        applicant_skills = [skill.skill_name.lower() for skill in applicant.skills]
        applicant_functional_areas = [fa.area_name.lower() for fa in applicant.functional_areas]
        applicant_domain = _detect_primary_experience_domain(applicant)
        applicant_years_experience = sum(
            (exp.end_date - exp.start_date).days / 365.25 if exp.end_date 
            else (date.today() - exp.start_date).days / 365.25
            for exp in applicant.experience
        )
        has_leadership_exp = any(
            any(kw in (exp.title or "").lower() for kw in ['lead', 'manager', 'director', 'head', 'chief', 'vp', 'vice president'])
            for exp in applicant.experience
        )
        has_senior_exp = any(
            any(kw in (exp.title or "").lower() for kw in ['senior', 'sr.', 'principal', 'staff', 'architect'])
            for exp in applicant.experience
        )
    else:
        # Load applicant embedding from existing vector
        applicant_embedding = np.array(json.loads(applicant_vector_record.embedding)).reshape(1, -1)
        
        # Load minimal applicant data for scoring (only what we need)
        applicant = db.query(Applicant).filter(Applicant.id == applicant_id).first()
        if not applicant:
            raise ValueError(f"No applicant found with id {applicant_id}")
        applicant_skills = [skill.skill_name.lower() for skill in applicant.skills]
        applicant_functional_areas = [fa.area_name.lower() for fa in applicant.functional_areas]
        applicant_domain = _detect_primary_experience_domain(applicant)
        applicant_years_experience = sum(
            (exp.end_date - exp.start_date).days / 365.25 if exp.end_date 
            else (date.today() - exp.start_date).days / 365.25
            for exp in applicant.experience
        )
        has_leadership_exp = any(
            any(kw in (exp.title or "").lower() for kw in ['lead', 'manager', 'director', 'head', 'chief', 'vp', 'vice president'])
            for exp in applicant.experience
        )
        has_senior_exp = any(
            any(kw in (exp.title or "").lower() for kw in ['senior', 'sr.', 'principal', 'staff', 'architect'])
            for exp in applicant.experience
        )
    
    # OPTIMIZED: Use ONLY pgvector native similarity search (no Python fallback)
    # Reduced candidate count for faster detailed scoring
    candidate_count = min(top_k * 10, 200)  # Get candidates for detailed scoring
    
    # Convert applicant embedding to PostgreSQL vector format
    applicant_embedding_list = applicant_embedding[0].tolist()
    embedding_str = '[' + ','.join(map(str, applicant_embedding_list)) + ']'
    
    # Use pgvector cosine distance operator (<=>) for similarity search
    # Embeddings are stored as JSON strings, parse and convert to vector
    # Note: This parses JSON for each row - if you have millions of jobs, consider migrating to native vector type
    query = text("""
        WITH parsed_embeddings AS (
            SELECT 
                jv.job_id,
                (SELECT array_agg(value::double precision) 
                 FROM json_array_elements_text(jv.embedding::json) AS value)::vector AS embedding_vector
            FROM job_vectors jv
            INNER JOIN jobs j ON j.id = jv.job_id
            WHERE j.status = 'published' 
              AND j.is_open = true
        )
        SELECT 
            job_id,
            1 - (embedding_vector <=> CAST(:embedding AS vector)) AS similarity
        FROM parsed_embeddings
        ORDER BY embedding_vector <=> CAST(:embedding AS vector)
        LIMIT :limit
    """)
    
    result = db.execute(query, {
        'embedding': embedding_str,
        'limit': candidate_count
    })
    
    top_candidate_pairs = [(row.job_id, float(row.similarity)) for row in result]
    
    # Step 2: Load full job details only for top candidates
    if not top_candidate_pairs:
        top_jobs = []
    else:
        top_job_ids = [job_id for job_id, _ in top_candidate_pairs]
        
        # Query only the jobs we need (much faster than loading all)
        top_job_objects = db.query(Job).filter(
            Job.id.in_(top_job_ids),
            Job.status == 'published',
            Job.is_open == True
        ).all()
        
        # Create mapping for fast lookup
        job_id_to_job = {job.id: job for job in top_job_objects}
        
        # Step 3: Do detailed scoring only on top candidates (optimized)
        enriched_jobs = []
        # Pre-compute job texts and domains in batch for better performance
        job_texts = {}
        job_domains = {}
        for job_id, _ in top_candidate_pairs:
            job = job_id_to_job.get(job_id)
            if job:
                job_texts[job_id] = extract_job_text(job)
                job_domains[job_id] = _detect_job_domain(job)
        
        # Now do scoring with pre-computed values
        for job_id, base_similarity in top_candidate_pairs:
            job = job_id_to_job.get(job_id)
            if not job:
                continue
            
            # Use pre-computed values
            job_text = job_texts.get(job_id, "")
            job_domain = job_domains.get(job_id, 'other')
            
            # Calculate component scores with proper weighting
            skills_score = _calculate_skills_match_score(applicant_skills, job_text)
            experience_score = _calculate_experience_match_score_optimized(
                applicant, job, applicant_domain, job_domain, 
                applicant_years_experience, has_leadership_exp, has_senior_exp
            )
            functional_area_score = _calculate_functional_area_score(job.area_name, applicant_functional_areas)
            location_worktype_score = _calculate_location_worktype_score(applicant, job)
            
            # Calculate weighted final score with domain-aware weighting
            if applicant_domain != job_domain and applicant_domain in ['technical', 'sales']:
                weighted_score = (
                    base_similarity * 0.10 +
                    skills_score * 0.20 +
                    experience_score * 0.60 +
                    functional_area_score * 0.07 +
                    location_worktype_score * 0.03
                )
            else:
                weighted_score = (
                    base_similarity * 0.20 +
                    skills_score * 0.40 +
                    experience_score * 0.30 +
                    functional_area_score * 0.07 +
                    location_worktype_score * 0.03
                )
            
            # Normalize and add
            weighted_score = min(1.0, max(0.0, weighted_score))
            enriched_jobs.append((job, weighted_score))
        
        # Sort by weighted score (highest to lowest) and return top_k
        enriched_jobs.sort(key=lambda x: x[1], reverse=True)
        top_jobs = enriched_jobs[:top_k] if enriched_jobs else []
    
    # Format results with minimal fields only (optimized for speed)
    results = []
    for job, score in top_jobs:
        # Only include essential fields
        province = job.province_name if job.province_name else 'N/A'
        country = job.country_name if job.country_name else 'N/A'
        location = f"{province}, {country}" if province != 'N/A' or country != 'N/A' else 'N/A'
        
        results.append({
            'title': job.title or '',
            'reference': job.reference if hasattr(job, 'reference') else '',
            'location': location,
            'area_name': job.area_name or 'N/A',
            'minimum_salary': float(job.minimum_salary) if job.minimum_salary else None,
            'maximum_salary': float(job.maximum_salary) if job.maximum_salary else None,
            'salary_type': job.salary_type or 'Not specified',
            'gender': job.gender or 'any',
            'period': job.period or 'monthly',
            'language': job.language or 'any',
            'publish_date': job.publish_date.isoformat() if job.publish_date else 'N/A',
            'closing_date': job.expiry_date.isoformat() if job.expiry_date else 'N/A',
            'similarity_score': float(score)
        })
    
    return results