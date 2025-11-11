#!/bin/bash
# Setup script for Job Recommendation System

set -e  # Exit on error

echo "🚀 Setting up Job Recommendation System..."

# Start the services
echo "🐳 Starting Docker containers..."
docker-compose up -d postgres

# Wait for postgres to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
        echo "✅ PostgreSQL is ready!"
        break
    fi
    attempt=$((attempt + 1))
    echo "   Attempt $attempt/$max_attempts..."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ PostgreSQL failed to start. Please check the logs."
    exit 1
fi

# Wait a bit more for pgvector extension to be available
sleep 3

# Verify pgvector extension
echo "🔍 Verifying pgvector extension..."
docker-compose exec -T postgres psql -U postgres -d job_recommender -c "CREATE EXTENSION IF NOT EXISTS vector;" > /dev/null 2>&1

# Run seeder
echo "🌱 Seeding database with 100 jobs and 5 applicants (demo mode)..."
docker-compose run --rm app python -c "
from seeder.seeder import seed_database
from db.database import SessionLocal
db = SessionLocal()
try:
    seed_database(db, num_jobs=100, num_applicants=5)
    print('✅ Database seeded successfully')
finally:
    db.close()
"

# Compute vectors
echo "🧮 Computing vectors for jobs and applicants..."
docker-compose run --rm app python -c "
from utils.vector_utils import compute_all_job_vectors, compute_all_applicant_vectors
from db.database import SessionLocal
db = SessionLocal()
try:
    print('Computing job vectors...')
    compute_all_job_vectors(db, batch_size=10)
    print('Computing applicant vectors...')
    compute_all_applicant_vectors(db, batch_size=5)
    print('✅ All vectors computed successfully')
finally:
    db.close()
"

# Start the application
echo "🔥 Starting FastAPI application..."
docker-compose up -d app

# Wait for API to be ready
echo "⏳ Waiting for API to be ready..."
sleep 5

# Check if API is running
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is running!"
else
    echo "⚠️  API might still be starting. Check logs with: docker-compose logs app"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Available endpoints:"
echo "   - GET  http://localhost:8000/health (health check)"
echo "   - GET  http://localhost:8000/docs (interactive API docs)"
echo "   - GET  http://localhost:8000/recommendations/{applicant_id}"
echo "   - GET  http://localhost:8000/recommendations/{applicant_id}/paginated?page=1&size=10"
echo "   - GET  http://localhost:8000/jobs?area_name=Software+Engineering&page=1&size=10"
echo "   - GET  http://localhost:8000/applicants?page=1&size=10"
echo ""
echo "💡 Quick start:"
echo "   1. Get an applicant ID: curl http://localhost:8000/applicants?page=1&size=1"
echo "   2. Get recommendations: curl http://localhost:8000/recommendations/{applicant_id}?top_k=20"
echo ""
echo "📊 The system is now running with:"
echo "   - 100 job listings"
echo "   - 5 applicant profiles"
echo "   - Precomputed embeddings for fast recommendations"
echo "   - Background vector computation scheduler"
echo "   - Paginated API endpoints"
echo ""
echo "📖 For more information, see README.md"