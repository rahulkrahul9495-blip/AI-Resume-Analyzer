"""AI Resume Analyzer - run with: streamlit run app.py"""
import os
import streamlit as st
from dotenv import load_dotenv
from analyzer.parser import extract_text
from analyzer.scoring import analyze
from analyzer.skills import group_by_category
from analyzer.ai_feedback import ai_review

load_dotenv()
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")
st.title("📄 AI Resume Analyzer")
st.caption("ATS-style scoring, job-description matching and AI coaching.")

with st.sidebar:
    st.header("Settings")
    use_ai = st.toggle("Enable Claude feedback", value=bool(os.getenv("ANTHROPIC_API_KEY")))
    key = st.text_input("Anthropic API key", type="password",
                        help="Or set ANTHROPIC_API_KEY in .env") if use_ai else None

col1, col2 = st.columns(2)
with col1:
    file = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"])
with col2:
    jd = st.text_area("Job description (optional)", height=180,
                      placeholder="Paste the job posting here for a match score...")

if st.button("Analyze", type="primary", disabled=file is None):
    try:
        text = extract_text(file.name, file.getvalue())
    except Exception as e:
        st.error(str(e))
        st.stop()

    rep = analyze(text, jd)
    st.divider()
    c1, c2 = st.columns([1, 3])
    c1.metric("Overall score", f"{rep.overall}/100")
    c1.progress(rep.overall / 100)
    with c2:
        cols = st.columns(len(rep.scores))
        for col, (name, val) in zip(cols, rep.scores.items()):
            col.metric(name, val)

    t1, t2, t3, t4 = st.tabs(["Improvements", "Skills", "Structure", "AI Review"])

    with t1:
        for tip in rep.tips or ["Looks solid. Nice work!"]:
            st.write("• " + tip)
        st.caption(f"{rep.word_count} words · {rep.action_verb_count} action verbs · "
                   f"{rep.quantified_bullets}/{rep.total_bullets} quantified bullets")

    with t2:
        if jd.strip():
            a, b = st.columns(2)
            a.success("Matched: " + (", ".join(rep.matched) or "none"))
            b.warning("Missing: " + (", ".join(rep.missing) or "none"))
            st.write(f"TF-IDF similarity to JD: **{rep.similarity:.0%}**")
        for cat, items in group_by_category(rep.resume_skills).items():
            st.write(f"**{cat}:** " + ", ".join(items))

    with t3:
        st.write("**Contact info**")
        st.write({k: "✅" if v else "❌" for k, v in rep.contact.items()})
        st.write("**Sections**")
        st.write({k: "✅" if v else "❌" for k, v in rep.sections.items()})

    with t4:
        if not use_ai:
            st.info("Enable Claude feedback in the sidebar.")
        else:
            with st.spinner("Claude is reviewing..."):
                try:
                    fb = ai_review(text, jd, key)
                except Exception as e:
                    st.error(f"AI review failed: {e}")
                    st.stop()
            st.write(fb.get("summary", ""))
            x, y = st.columns(2)
            x.subheader("Strengths")
            for s in fb.get("strengths", []): x.write("✅ " + s)
            y.subheader("Weaknesses")
            for s in fb.get("weaknesses", []): y.write("⚠️ " + s)
            if fb.get("rewrites"):
                st.subheader("Suggested rewrites")
                for rw in fb["rewrites"]:
                    st.markdown(f"~~{rw['original']}~~\n\n**→ {rw['improved']}**")
            if fb.get("role_fit"):
                st.subheader("Role fit")
                st.write(fb["role_fit"])
