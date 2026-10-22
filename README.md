
# ResumeMatch AI — AI Resume & Job Matcher

A beginner-friendly Streamlit app that compares a PDF resume with a
pasted job description using Google's Gemini API. It estimates relevance
and suggests truthful, practical ways to improve a resume.

> Important: The match score is an AI estimate, not an official ATS score,
> hiring decision, or guarantee of employment. Always verify suggestions
> and never add skills or experience you do not have.

## Features

- PDF upload with file type, size (5 MB), and page-count (25 pages) limits.
- PDF text extraction with pypdf.
- Detection of scanned/image-only PDFs (OCR is not included).
- Estimated match score from 0–100.
- Pandas-powered skills comparison table for matching and missing skills.
- Interactive Plotly bar and donut charts for skills analytics.
- Dashboard metrics for match score, matching skills, and skills to develop.
- Matching skills, missing skills, strengths, resume improvements,
  relevant keywords, and interview preparation suggestions.
- Structured JSON output validated with Pydantic before display.
- Friendly handling for missing keys, invalid PDFs, empty inputs,
  rate limits, and service errors.
- No database or application logging of resume contents.

Resume text and job descriptions are sent to Google Gemini for analysis.
Upload only documents you are permitted to share and review the applicable
provider terms and privacy settings.

## Requirements

- Python 3.10 or newer
- A Google Gemini API key with access to the selected model
- A GitHub account
- A Streamlit Community Cloud account

## Run locally

### 1. Open the project folder

Download or clone this repository and open a terminal in that folder.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the API key

Create a folder named `.streamlit` in the project root.
Inside it, create a file named `secrets.toml` containing:

```toml
GEMINI_API_KEY = "paste_your_own_key_here"
```

Get a key through Google's official Gemini API key page:

https://aistudio.google.com/app/apikey

Replace the example value with your own key. Keep the key private.
The `.gitignore` excludes `.streamlit/secrets.toml`.

Alternative for local development: set the environment variable
`GEMINI_API_KEY`.

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="paste_your_own_key_here"
```

macOS/Linux:

```bash
export GEMINI_API_KEY="paste_your_own_key_here"
```

### 5. Start the app

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, usually:

http://localhost:8501

## Upload to GitHub using your browser

1. Sign in to GitHub and select New repository.
2. Name it `resume-match-ai`.
3. Choose Public or Private, then create the repository.
4. You can skip creating a README online because this project already has one.
5. On the repository page, select Add file → Upload files.
6. Upload these four files:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
7. Do NOT upload `.streamlit/secrets.toml`, `.env`, or any real API key.
8. Enter a commit message such as `Add ResumeMatch AI application`.
9. Commit the files.

## Alternative: upload with Git commands

Run these commands from the project folder after creating an empty
GitHub repository. Replace the URL with your own repository URL.

```bash
git init
git add app.py requirements.txt README.md .gitignore
git commit -m "Add ResumeMatch AI application"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/resume-match-ai.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username.

## Deploy on Streamlit Community Cloud

1. Push all four project files to GitHub.
2. Sign in at https://share.streamlit.io/
3. Connect your GitHub account if prompted.
4. Select Create app.
5. Choose your repository and branch (usually `main`).
6. Set the main file path to `app.py`.
7. Open the app's Settings → Secrets section.
8. Add the following TOML secret, replacing the example with your own key:

   ```toml
   GEMINI_API_KEY = "paste_your_own_key_here"
   ```

9. Save the secret and deploy or reboot the app if necessary.
10. Open the deployed app and test it with a small text-based PDF
    and a real job description.

Streamlit Community Cloud installs the packages in `requirements.txt`.
If deployment fails, check the build/runtime logs for dependency or
configuration errors. Never share your API key in screenshots or logs.

## How it works

1. The user uploads a PDF and pastes a job description.
2. The app validates the upload and extracts selectable text with pypdf.
3. The app checks for an API key and sends the resume text and job
   description to Gemini.
4. Gemini returns structured JSON, which Pydantic validates.
5. The app displays the score, matching skills, gaps, recommendations,
   and interview preparation suggestions.

The app does not deliberately persist resume text to a database or
application logs. Provider and hosting operational processing may still apply.

## Limitations and privacy notes

- Scanned PDFs are not supported in this version; OCR is not included.
- PDF formatting may not be preserved during text extraction.
- AI results can be inaccurate. Verify every claim and recommendation.
- Resume content and job descriptions are sent to Gemini for analysis.
- Do not upload sensitive documents unless you are comfortable with the
  applicable provider terms.
- The application does not save resumes to a database or log their contents.
  This is not a guarantee that hosting or API providers retain no operational data.

## Model and SDK

The app uses the configured Gemini model identifier in `app.py` and Google's
official `google-genai` Python SDK. Pandas organizes the skills data into tables,
and Plotly Express renders the interactive analytics charts.

Check the official Gemini model list if availability changes:

https://ai.google.dev/gemini-api/docs/models
