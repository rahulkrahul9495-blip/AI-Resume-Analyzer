"""Rule-based + statistical resume scoring."""
import re
from dataclasses import dataclass, field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .skills import extract_skills

SECTION_PATTERNS = {
    "Summary": r"^(summary|profile|objective|about me)\b",
    "Experience": r"^(work |professional )?(experience|employment|work history)\b",
    "Education": r"^(education|academic)\b",
    "Skills": r"^(technical )?skills\b",
    "Projects": r"^(projects|personal projects)\b",
    "Certifications": r"^(certifications?|licenses?|courses)\b",
}

ACTION_VERBS = set("""achieved analyzed architected automated built collaborated created
delivered designed developed deployed drove enhanced established executed implemented improved
increased integrated launched led managed mentored migrated optimized orchestrated reduced
refactored resolved scaled shipped streamlined supervised trained transformed""".split())

WEAK_PHRASES = ["responsible for", "worked on", "helped with", "duties included",
                "tasked with", "assisted in", "participated in"]

CONTACT_PATTERNS = {
    "Email": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "Phone": r"(\+?\d[\d\s().-]{8,}\d)",
    "LinkedIn": r"linkedin\.com/in/[\w-]+",
    "GitHub": r"github\.com/[\w-]+",
}


@dataclass
class Report:
    overall: int = 0
    scores: dict = field(default_factory=dict)
    contact: dict = field(default_factory=dict)
    sections: dict = field(default_factory=dict)
    resume_skills: dict = field(default_factory=dict)
    matched: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    similarity: float | None = None
    word_count: int = 0
    action_verb_count: int = 0
    quantified_bullets: int = 0
    total_bullets: int = 0
    weak_phrases: list = field(default_factory=list)
    tips: list = field(default_factory=list)


def detect_sections(text: str) -> dict:
    found = {k: False for k in SECTION_PATTERNS}
    for line in text.splitlines():
        line = line.strip().lower().rstrip(":")
        if 0 < len(line) < 40:
            for name, pat in SECTION_PATTERNS.items():
                if re.match(pat, line):
                    found[name] = True
    return found


def bullets(text: str) -> list[str]:
    return [l.strip("-*• \t") for l in text.splitlines()
            if re.match(r"^\s*[-*•]", l) and len(l.split()) > 3]


def jd_similarity(resume: str, jd: str) -> float:
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    m = vec.fit_transform([resume, jd])
    return float(cosine_similarity(m[0], m[1])[0][0])


def analyze(resume: str, jd: str = "") -> Report:
    r = Report()
    low = resume.lower()
    words = re.findall(r"[A-Za-z']+", resume)
    r.word_count = len(words)

    # Contact
    r.contact = {k: bool(re.search(p, resume, re.I)) for k, p in CONTACT_PATTERNS.items()}
    contact_score = 100 * sum(r.contact.values()) / len(r.contact)

    # Sections
    r.sections = detect_sections(resume)
    core = ["Experience", "Education", "Skills"]
    section_score = (70 * sum(r.sections[s] for s in core) / 3
                     + 30 * sum(r.sections.values()) / len(r.sections))

    # Skills
    r.resume_skills = extract_skills(resume)
    skills_score = min(100, len(r.resume_skills) * 8)

    # Impact: verbs, metrics, weak phrases
    r.action_verb_count = sum(1 for w in words if w.lower() in ACTION_VERBS)
    bl = bullets(resume)
    r.total_bullets = len(bl)
    r.quantified_bullets = sum(1 for b in bl if re.search(r"\d+%?|\$\d+|\d+x", b))
    r.weak_phrases = [p for p in WEAK_PHRASES if p in low]
    verb_score = min(100, r.action_verb_count * 10)
    quant_ratio = (r.quantified_bullets / r.total_bullets) if r.total_bullets else 0
    impact_score = 0.4 * verb_score + 0.6 * min(100, quant_ratio * 200) - 5 * len(r.weak_phrases)
    impact_score = max(0, impact_score)

    # Length (ideal 350-800 words)
    if 350 <= r.word_count <= 800:
        length_score = 100
    elif r.word_count < 350:
        length_score = max(20, r.word_count / 350 * 100)
    else:
        length_score = max(40, 100 - (r.word_count - 800) / 10)

    r.scores = {"Contact": contact_score, "Sections": section_score, "Skills": skills_score,
                "Impact": impact_score, "Length": length_score}
    weights = {"Contact": .10, "Sections": .20, "Skills": .20, "Impact": .30, "Length": .20}

    # Job description match
    if jd.strip():
        jd_skills = extract_skills(jd)
        r.matched = sorted(set(jd_skills) & set(r.resume_skills))
        r.missing = sorted(set(jd_skills) - set(r.resume_skills))
        kw = len(r.matched) / len(jd_skills) * 100 if jd_skills else 0
        r.similarity = jd_similarity(resume, jd)
        r.scores["JD Match"] = 0.6 * kw + 0.4 * min(100, r.similarity * 250)
        weights = {k: v * 0.6 for k, v in weights.items()}
        weights["JD Match"] = 0.4

    r.scores = {k: round(v) for k, v in r.scores.items()}
    r.overall = round(sum(r.scores[k] * w for k, w in weights.items()))
    r.tips = _tips(r)
    return r


def _tips(r: Report) -> list[str]:
    t = []
    for k, ok in r.contact.items():
        if not ok:
            t.append(f"Add your {k} to the header.")
    for s in ["Experience", "Education", "Skills"]:
        if not r.sections[s]:
            t.append(f"Add a clearly labelled '{s}' section so ATS parsers find it.")
    if r.total_bullets and r.quantified_bullets / r.total_bullets < 0.4:
        t.append("Quantify more bullets (%, $, time saved, users, scale). Aim for 40%+.")
    if r.action_verb_count < 6:
        t.append("Start bullets with strong action verbs (built, led, optimized).")
    if r.weak_phrases:
        t.append("Replace weak phrases: " + ", ".join(f"'{p}'" for p in r.weak_phrases) + ".")
    if r.word_count < 350:
        t.append("Resume looks thin; expand projects and achievements.")
    elif r.word_count > 800:
        t.append("Resume is long; trim to the most relevant 1-2 pages.")
    if r.missing:
        t.append("Missing JD keywords (add only if truthful): " + ", ".join(r.missing[:10]) + ".")
    return t
