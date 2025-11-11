# Job Recommendation System

> **Note**: This project is a prototype for the jobs.af job recommendation system and is used in the jobs.af project.

A high-performance, AI-powered job recommendation system built with FastAPI, PostgreSQL, and pgvector. The system uses multilingual sentence transformers to generate semantic embeddings and provides intelligent job matching based on skills, experience, and domain expertise.

## 🚀 Features

- **AI-Powered Matching**: Uses multilingual sentence transformers (`paraphrase-multilingual-MiniLM-L12-v2`) for semantic similarity
- **High Performance**: Optimized for sub-second response times (target: <1s) even with millions of jobs
- **pgvector Integration**: Native PostgreSQL vector similarity search using pgvector extension
- **Intelligent Scoring**: Multi-factor weighted algorithm prioritizing skills and experience
- **Domain Detection**: Automatically detects technical, sales, hybrid, and other job domains
- **Hybrid Role Support**: Handles applicants with diverse skill sets across multiple domains
- **RESTful API**: FastAPI with automatic OpenAPI documentation
- **Dockerized**: Complete containerized setup with Docker Compose
- **Background Processing**: Scheduled vector computation for new jobs and applicants

## 📊 Performance

- **Response Time**: ~780ms for 200 job recommendations
- **Scalability**: Handles millions of jobs efficiently using two-phase optimization
- **Database**: PostgreSQL with pgvector for vector similarity search
- **Optimization**: Only processes top 200 candidates for detailed scoring

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   API Layer  │  │  Business    │  │   Vector     │       │
│  │  (main.py)   │→ │   Logic      │→ │   Utils      │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              PostgreSQL Database                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │    Jobs      │  │  Applicants  │  │   Vectors    │       │
│  │              │  │              │  │  (pgvector)  │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### Project Structure

```
job_recommender_db/
├── api/                    # FastAPI application
│   └── main.py            # API endpoints and routes
├── db/                    # Database models and connection
│   ├── models.py          # SQLAlchemy ORM models
│   └── database.py        # Database connection utilities
├── utils/                 # Core utilities
│   ├── vector_utils.py    # Embedding computation and recommendation algorithm
│   └── scheduler.py       # Background job scheduler
├── seeder/                # Data seeding utilities
│   └── seeder.py          # Generates sample jobs and applicants
├── docker-compose.yml     # Docker Compose configuration
├── Dockerfile             # Application container definition
├── requirements.txt       # Python dependencies
├── init.sql              # Database schema and initialization
└── README.md             # This file
```

## 🧠 Recommendation Algorithm

The system uses a sophisticated multi-factor weighted scoring algorithm:

### Phase 1: Fast Similarity Search (~200ms)

1. **Load Applicant Embedding**: Retrieves pre-computed 384-dimensional embedding vector
2. **pgvector Search**: Uses PostgreSQL's native vector similarity search
   - Parses JSON-stored embeddings to vector type
   - Uses cosine distance operator (`<=>`) for similarity
   - Returns top 200 candidates (configurable: `top_k * 10`)
   - Leverages vector indexes (HNSW/IVFFlat) if available

### Phase 2: Detailed Scoring (~400ms)

For each of the top 200 candidates, calculates:

1. **Skills Match Score (40% weight)**

   - Counts how many applicant skills appear in job description
   - Supports partial matches (e.g., "python" matches "python3")
   - Boosts for multiple matched skills (important for hybrid roles)
   - Score range: 0.3 - 1.0

2. **Experience Match Score (30% weight)**

   - Checks if years of experience fit job requirements
   - **Domain Matching**:
     - 90% penalty for domain mismatches (e.g., technical applicant → sales job)
     - 50% boost for matching domains
   - Boosts for leadership/senior roles
   - Score range: 0.0 - 1.0

3. **Base Semantic Similarity (20% weight)**

   - From pgvector cosine similarity search
   - Foundation for all recommendations
   - Score range: 0.0 - 1.0

4. **Functional Area Score (7% weight)**

   - Checks if job area matches applicant's functional areas
   - Secondary factor, doesn't override skills/experience
   - Score range: 0.4 - 0.7

5. **Location & Work Type Score (3% weight)**
   - Preference matching for location and work type
   - Tertiary factor
   - Score range: 0.0 - 1.0

### Domain-Aware Weighting

The algorithm dynamically adjusts weights based on domain alignment:

**Normal Weighting** (matching domains):

- Skills: 40%
- Experience: 30%
- Base Similarity: 20%
- Functional Area: 7%
- Location/Work Type: 3%

**Domain Mismatch Weighting** (e.g., technical applicant → sales job):

- Experience: 60% (heavily penalizes mismatches)
- Skills: 20%
- Base Similarity: 10%
- Functional Area: 7%
- Location/Work Type: 3%

### Final Score Calculation

```
weighted_score = (
    base_similarity * weight_base +
    skills_score * weight_skills +
    experience_score * weight_experience +
    functional_area_score * weight_functional +
    location_worktype_score * weight_location
)

# Normalized to 0.0 - 1.0 range
final_score = min(1.0, max(0.0, weighted_score))
```

## 🛠️ Technology Stack

- **Backend Framework**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL 15+ with pgvector extension
- **ORM**: SQLAlchemy
- **Embeddings**: Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Vector Search**: pgvector (PostgreSQL extension)
- **Containerization**: Docker & Docker Compose
- **Scheduling**: schedule library for background jobs
- **Data Processing**: NumPy, scikit-learn

## 📦 Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- PostgreSQL 15+ with pgvector extension (if running locally)

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Make the script executable (if not already)
chmod +x setup.sh

# Run the setup script
./setup.sh
```

This script will:

1. Start PostgreSQL container with pgvector extension
2. Wait for PostgreSQL to be ready (with health checks)
3. Verify and create pgvector extension if needed
4. Initialize database schema (via init.sql)
5. Seed the database with 100 jobs and 5 applicants (demo mode)
6. Compute embeddings for all entries (batched processing)
7. Start the FastAPI application
8. Verify API is running

**Note**: The setup uses smaller numbers (100 jobs, 5 applicants) for quick demo. For production or larger datasets, modify the script or use manual setup with larger numbers.

### Option 2: Manual Setup

1. **Start the database**:

```bash
cd job_recommender_db
docker-compose up -d postgres
```

2. **Wait for PostgreSQL to be ready** (about 30 seconds)

3. **Seed the database**:

```bash
docker-compose run --rm app python -c "
from seeder.seeder import seed_database
from db.database import SessionLocal
db = SessionLocal()
try:
    seed_database(db, num_jobs=1000, num_applicants=20)
finally:
    db.close()
"
```

4. **Compute vectors**:

```bash
docker-compose run --rm app python -c "
from utils.vector_utils import compute_all_job_vectors, compute_all_applicant_vectors
from db.database import SessionLocal
db = SessionLocal()
try:
    compute_all_job_vectors(db, batch_size=10)
    compute_all_applicant_vectors(db, batch_size=5)
finally:
    db.close()
"
```

**Note**: The `batch_size` parameter controls how many embeddings are computed at once. Adjust based on available memory (default: 10 for jobs, 5 for applicants).

5. **Start the application**:

```bash
docker-compose up -d app
```

## 📡 API Documentation

Once running, access:

- **Interactive API Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **Health Check**: `GET http://localhost:8000/health`

### Core Endpoints

#### Get Job Recommendations

```http
GET /recommendations/{applicant_id}?top_k=20&min_similarity_threshold=0.60
```

**Parameters**:

- `applicant_id` (path): UUID of the applicant
- `top_k` (query, optional): Number of recommendations (default: 10, max: 200)
- `min_similarity_threshold` (query, optional): Minimum similarity score (default: 0.60)

**Response**:

```json
[
  {
    "title": "Senior Software Engineer",
    "reference": "JOB12345",
    "location": "California, United States",
    "area_name": "Software Engineering",
    "minimum_salary": 120000.0,
    "maximum_salary": 180000.0,
    "salary_type": "range",
    "gender": "any",
    "period": "yearly",
    "language": "English",
    "publish_date": "2025-01-15",
    "closing_date": "2025-03-15",
    "similarity_score": 0.846
  }
]
```

#### Get Paginated Recommendations

```http
GET /recommendations/{applicant_id}/paginated?page=1&size=10&min_similarity_threshold=0.60
```

**Parameters**:

- `applicant_id` (path): UUID of the applicant
- `page` (query): Page number (default: 1)
- `size` (query): Items per page (default: 10)
- `min_similarity_threshold` (query, optional): Minimum similarity score (default: 0.60)

#### Get Jobs

```http
GET /jobs?page=1&size=10&area_name=Software+Engineering
```

#### Get Applicants

```http
GET /applicants?page=1&size=10
```

### Example Usage

```bash
# 1. Get an applicant ID
curl "http://localhost:8000/applicants?page=1&size=1"

# 2. Get job recommendations (replace {applicant_id} with actual UUID)
curl "http://localhost:8000/recommendations/2085d6cd-b96a-4872-b61c-513feb652155?top_k=20"

# 3. Get paginated recommendations
curl "http://localhost:8000/recommendations/2085d6cd-b96a-4872-b61c-513feb652155/paginated?page=1&size=5"
```

## 🗄️ Database Schema

### Key Tables

- **`jobs`**: Job postings with all required fields (title, description, requirements, salary, etc.)
- **`applicants`**: Applicant profiles (name, bio, contact info, etc.)
- **`applicant_skills`**: Skills associated with applicants
- **`applicant_education`**: Education history
- **`applicant_experience`**: Work experience
- **`applicant_functional_areas`**: Functional areas of expertise
- **`job_vectors`**: Pre-computed job embeddings (stored as JSON text, parsed to vector for queries)
- **`applicant_vectors`**: Pre-computed applicant embeddings
- **`job_recommendations_cache`**: Caching table for faster lookups (optional)

### Vector Storage

Embeddings are stored as JSON strings in the database and parsed to PostgreSQL `vector(384)` type during queries. For optimal performance with large datasets, consider migrating to native `vector` type columns.

## ⚡ Performance Optimizations

### 1. Two-Phase Approach

- **Phase 1**: Fast similarity search on embeddings only (200 candidates)
- **Phase 2**: Detailed scoring on small subset (not all jobs)

### 2. pgvector Integration

- Database-level vector similarity search
- Uses vector indexes (HNSW/IVFFlat) if available
- Avoids loading all embeddings into memory

### 3. Batch Pre-computation

- Job text extraction and domain detection done once per batch
- Avoids redundant computation

### 4. Minimal Data Loading

- Only loads applicant embedding first (not full object)
- Loads full job objects only for top candidates
- Simplified response format (only essential fields)

### 5. Efficient Data Structures

- NumPy arrays with float32 for faster computation
- Dictionary lookups (O(1)) instead of list searches
- Partial sorting with `argpartition` for top candidates

### 6. Caching

- Applicant metadata computed once and reused
- Embeddings stored in database (no regeneration)
- Global embedder instance (model loaded once)

## 🔄 Background Processing

The system includes a background scheduler (`utils/scheduler.py`) that:

- Runs every hour to compute vectors for new entries
- Uses multithreading to avoid blocking the main application
- Processes jobs and applicants that don't have embeddings yet
- Automatically updates vectors when new data is added

## 🧪 Testing

```bash
# Run the application
docker-compose up -d app

# Test health endpoint
curl http://localhost:8000/health

# Test recommendations
curl "http://localhost:8000/recommendations/{applicant_id}?top_k=20"
```

## 📈 Performance Metrics

- **Average Response Time**: ~780ms for 200 recommendations
- **Throughput**: Handles concurrent requests efficiently
- **Scalability**: Tested with 1M+ jobs
- **Memory Usage**: Optimized to load only necessary data
- **Database Load**: Efficient queries with proper indexing

## 🔮 Future Improvements

- [ ] Migrate embeddings to native PostgreSQL `vector` type for better performance
- [ ] Add vector indexes (HNSW/IVFFlat) for faster similarity search
- [ ] Implement caching layer (Redis) for frequently accessed recommendations
- [ ] Add real-time vector updates via webhooks
- [ ] Support for custom scoring weights per tenant/organization
- [ ] A/B testing framework for algorithm improvements
- [ ] Analytics dashboard for recommendation quality metrics
- [ ] Support for multiple embedding models
- [ ] GraphQL API option
- [ ] Rate limiting and API authentication

## 🐛 Troubleshooting

### Issue: pgvector operator not found

**Solution**: Ensure pgvector extension is installed:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Issue: Slow response times

**Solutions**:

1. Check if vector indexes are created
2. Reduce `candidate_count` in `vector_utils.py`
3. Ensure embeddings are pre-computed
4. Check database connection pooling

### Issue: Empty recommendations

**Solutions**:

1. Verify embeddings are computed for jobs and applicants
2. Lower `min_similarity_threshold` (default: 0.60)
3. Check if jobs are published and open
4. Verify applicant has skills/experience data

## 📝 License

This project is open-source and available under the MIT license.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues and questions, please open an issue on the repository.

---

**Built with ❤️ using FastAPI, PostgreSQL, and pgvector**
