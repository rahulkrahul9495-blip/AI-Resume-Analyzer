"""Skill taxonomy and extraction. Extend SKILLS to fit your domain."""
import re

SKILLS = {
    "Languages": ["python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
                  "kotlin", "swift", "php", "ruby", "scala", "r", "sql", "bash", "matlab"],
    "Web": ["react", "angular", "vue", "node.js", "express", "django", "flask", "fastapi",
            "spring boot", "html", "css", "tailwind", "next.js", "rest", "graphql"],
    "Data & ML": ["machine learning", "deep learning", "nlp", "computer vision", "pandas",
                  "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "spark",
                  "hadoop", "tableau", "power bi", "data analysis", "statistics",
                  "llm", "langchain", "hugging face", "xgboost", "etl"],
    "Cloud & DevOps": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform",
                       "jenkins", "ci/cd", "github actions", "linux", "ansible", "git"],
    "Databases": ["mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle",
                  "elasticsearch", "dynamodb", "cassandra", "firebase"],
    "Soft Skills": ["leadership", "communication", "teamwork", "problem solving",
                    "project management", "agile", "scrum", "mentoring", "collaboration"],
}

ALIASES = {"js": "javascript", "ts": "typescript", "k8s": "kubernetes",
           "postgres": "postgresql", "sklearn": "scikit-learn", "node": "node.js",
           "reactjs": "react", "ml": "machine learning", "dl": "deep learning"}

_ALL = {s: cat for cat, items in SKILLS.items() for s in items}


def _pattern(skill: str) -> re.Pattern:
    # Custom boundaries so "c++", "c#", "node.js" and "r" match safely.
    return re.compile(r"(?<![A-Za-z0-9+#])" + re.escape(skill) + r"(?![A-Za-z0-9+#])",
                      re.IGNORECASE)


_PATTERNS = {s: _pattern(s) for s in _ALL}
_ALIAS_PATTERNS = {a: _pattern(a) for a in ALIASES}


def extract_skills(text: str) -> dict:
    """Return {skill: category} for every skill found in text."""
    found = {s: _ALL[s] for s, p in _PATTERNS.items() if p.search(text)}
    for alias, target in ALIASES.items():
        if _ALIAS_PATTERNS[alias].search(text):
            found[target] = _ALL[target]
    return found


def group_by_category(skills: dict) -> dict:
    out = {}
    for skill, cat in skills.items():
        out.setdefault(cat, []).append(skill)
    return {c: sorted(v) for c, v in out.items()}
