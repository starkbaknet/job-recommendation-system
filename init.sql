-- Database initialization script
-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Create jobs table
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    reference VARCHAR(50) UNIQUE NOT NULL,
    number_of_vacancies INTEGER DEFAULT 1,
    status VARCHAR(20) DEFAULT 'draft',
    is_open BOOLEAN DEFAULT TRUE,
    job_requirements TEXT,
    role_summary TEXT,
    duties_and_responsibilities TEXT,
    minimum_experience INTEGER,
    maximum_experience INTEGER,
    education_level VARCHAR(50),
    salary_type VARCHAR(20),
    minimum_salary DECIMAL(10, 2),
    maximum_salary DECIMAL(10, 2),
    currency VARCHAR(10) DEFAULT 'USD',
    period VARCHAR(20) DEFAULT 'monthly',
    work_type VARCHAR(20) DEFAULT 'full_time',
    gender VARCHAR(20) DEFAULT 'any',
    nationality VARCHAR(100),
    language VARCHAR(100),
    publish_date DATE,
    expiry_date DATE,
    company_id UUID,
    company_name VARCHAR(255),
    company_is_public BOOLEAN DEFAULT FALSE,
    company_work_policy VARCHAR(20) DEFAULT 'hybrid',  -- remote, on-site, hybrid
    industry_type_id VARCHAR(50),
    industry_type_name VARCHAR(100),
    country_code VARCHAR(10),
    country_name VARCHAR(100),
    province_name VARCHAR(100),
    area_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create applicants table
CREATE TABLE applicants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    phone VARCHAR(50),
    bio TEXT,
    nationality VARCHAR(100),
    date_of_birth DATE,
    gender VARCHAR(20),
    address TEXT,
    country_code VARCHAR(10),
    country_name VARCHAR(100),
    province_name VARCHAR(100),
    speaking_languages TEXT, -- JSON array of languages
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create applicant skills table
CREATE TABLE applicant_skills (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    applicant_id UUID REFERENCES applicants(id) ON DELETE CASCADE,
    skill_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create applicant functional areas table
CREATE TABLE applicant_functional_areas (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    applicant_id UUID REFERENCES applicants(id) ON DELETE CASCADE,
    area_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create applicant education table
CREATE TABLE applicant_education (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    applicant_id UUID REFERENCES applicants(id) ON DELETE CASCADE,
    level VARCHAR(50) NOT NULL, -- high_school, associate, bachelor, master, phd
    institute_name VARCHAR(255) NOT NULL,
    field_of_study VARCHAR(255),
    start_date DATE,
    end_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create applicant experience table
CREATE TABLE applicant_experience (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    applicant_id UUID REFERENCES applicants(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    company_name VARCHAR(255) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create applicant awards table
CREATE TABLE applicant_awards (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    applicant_id UUID REFERENCES applicants(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    issued_by VARCHAR(255) NOT NULL,
    issued_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create applicant certificates table
CREATE TABLE applicant_certificates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    applicant_id UUID REFERENCES applicants(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    issued_by VARCHAR(255) NOT NULL,
    issued_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create job vectors table for storing embeddings
CREATE TABLE job_vectors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID UNIQUE REFERENCES jobs(id) ON DELETE CASCADE,
    embedding vector(384), -- Using pgvector for embedding storage (384-dim from MiniLM)
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create applicant vectors table for storing embeddings
CREATE TABLE applicant_vectors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    applicant_id UUID UNIQUE REFERENCES applicants(id) ON DELETE CASCADE,
    embedding vector(384), -- Using pgvector for embedding storage (384-dim from MiniLM)
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create job recommendations cache table
CREATE TABLE job_recommendations_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    applicant_id UUID REFERENCES applicants(id) ON DELETE CASCADE,
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    similarity_score DECIMAL(5, 4),
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(applicant_id, job_id)
);

-- Create indexes for better performance
CREATE INDEX idx_jobs_status ON jobs(status) WHERE is_open = true;
CREATE INDEX idx_jobs_area ON jobs(area_name);
CREATE INDEX idx_jobs_publish_date ON jobs(publish_date);
CREATE INDEX idx_applicants_country ON applicants(country_code);
CREATE INDEX idx_applicant_skills_applicant_id ON applicant_skills(applicant_id);
CREATE INDEX idx_applicant_functional_areas_applicant_id ON applicant_functional_areas(applicant_id);
CREATE INDEX idx_job_vectors_job_id ON job_vectors(job_id);
CREATE INDEX idx_applicant_vectors_applicant_id ON applicant_vectors(applicant_id);

-- Add extension for vector operations if available
-- This assumes you have pgvector installed in your Postgres image
-- CREATE EXTENSION IF NOT EXISTS vector;