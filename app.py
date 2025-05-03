import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI-Based Resume Screening & Ranking App")

# Upload CSV
uploaded_file = st.file_uploader("Upload your resume dataset (.csv)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Normalize column names
    df.columns = [col.strip().lower() for col in df.columns]

    # Check for required columns
    required_cols = ['name', 'skills', 'experience (years)', 'education']
    if not all(col in df.columns for col in required_cols):
        st.error("Missing one or more required columns: 'Name', 'Skills', 'Experience (Years)', 'Education'")
    else:
        # Combine columns into a single profile text
        df['profile'] = df['skills'].astype(str) + ' ' + df['experience (years)'].astype(str) + ' ' + df['education'].astype(str)

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df['profile'])

        # Job Descriptions
        job_descriptions = {
            "Data Scientist": "Machine learning, Python, data analysis, cloud computing, statistics",
            "Frontend Developer": "HTML, CSS, JavaScript, React, Angular, UI/UX",
            "Backend Developer": "Node.js, Django, REST APIs, SQL, Docker",
            "DevOps Engineer": "CI/CD, cloud services, infrastructure as code, monitoring tools"
        }

        selected_job = st.selectbox("Select Job Role", list(job_descriptions.keys()))
        job_desc = job_descriptions[selected_job]

        # Rank candidates
        job_vec = vectorizer.transform([job_desc])
        scores = cosine_similarity(job_vec, tfidf_matrix).flatten()
        df['Match Score'] = scores
        top_matches = df.sort_values(by='Match Score', ascending=False).head(10)

        st.subheader("Top Matching Candidates")

        # Display with progress bars
        for _, row in top_matches.iterrows():
            st.write(f"**{row['name']}** - {row['education']}")
            st.write(f"Skills: {row['skills']}")
            st.write(f"Experience: {row['experience (years)']} years")
            percent = int(row['Match Score'] * 100)
            st.progress(row['Match Score'])
            st.write(f"**Match Score: {percent}%**")
            st.markdown("---")

        # Download
        csv = top_matches.to_csv(index=False).encode('utf-8')
        st.download_button("Download Top Matches CSV", csv, f"top_matches_{selected_job.replace(' ', '_')}.csv", "text/csv")

        # Optional: show full table
        with st.expander("View Full Table"):
            st.dataframe(top_matches)