"""LLM feedback via the Anthropic API. Returns structured JSON."""
import json
import os
import re
import anthropic

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

SYSTEM = """You are a senior technical recruiter and resume coach.
Return ONLY valid JSON with this schema, no markdown:
{"summary": str,
 "strengths": [str],
 "weaknesses": [str],
 "rewrites": [{"original": str, "improved": str}],
 "role_fit": str}
Rules: never invent experience or numbers; where a metric is missing, use a
placeholder like [X%] in rewrites. Give at most 4 rewrites."""


def ai_review(resume: str, jd: str = "", api_key: str | None = None) -> dict:
    client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
    prompt = f"<resume>\n{resume[:12000]}\n</resume>"
    if jd.strip():
        prompt += f"\n<job_description>\n{jd[:6000]}\n</job_description>"
    msg = client.messages.create(
        model=MODEL, max_tokens=1500, system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in msg.content if b.type == "text")
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"summary": raw, "strengths": [], "weaknesses": [], "rewrites": [], "role_fit": ""}
