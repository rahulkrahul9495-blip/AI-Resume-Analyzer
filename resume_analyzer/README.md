# AI Resume Analyzer

Upload a resume (PDF/DOCX/TXT), optionally paste a job description, and get:
- Overall score with sub-scores (contact, sections, skills, impact, length, JD match)
- Skill extraction + missing-keyword detection
- TF-IDF cosine similarity to the job description
- Optional Claude feedback: strengths, weaknesses, bullet rewrites

## Run
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env     # add your ANTHROPIC_API_KEY
    streamlit run app.py

## Structure
    app.py                  Streamlit UI
    analyzer/parser.py      PDF/DOCX/TXT text extraction
    analyzer/skills.py      Skill taxonomy and matcher
    analyzer/scoring.py     Scoring engine and tips
    analyzer/ai_feedback.py Claude API integration
