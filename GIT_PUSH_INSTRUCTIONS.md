# How to Push cosmos-mas to GitHub
# ═══════════════════════════════════════════════════════════════

## Step 1 — Create the GitHub repository

1. Go to https://github.com/new
2. Repository name: cosmos-mas
3. Description: COSMOS: Collaborative Strategy Meta-Optimization via Self-Reflective Restructuring (NeurIPS 2026)
4. Set to PUBLIC
5. Do NOT initialise with README (you already have one)
6. Click "Create repository"

## Step 2 — Set up Git locally (run these commands in your terminal)

```bash
# Navigate to the repo folder
cd cosmos-mas

# Initialise git
git init

# Add your identity (use your real name/email)
git config user.name  "Aditya Diwan"
git config user.email "2302030430127@silveroakuni.ac.in"

# Stage all files
git add .

# First commit
git commit -m "Initial release: COSMOS NeurIPS 2026

- Full implementation: hypergraph, quality, CTP, validation, system
- 63 tests passing (pytest)
- DistributedCodeBench benchmark harness
- All system prompts
- Reproducibility checklist
- MIT License"

# Connect to GitHub (replace YOUR_GITHUB_USERNAME if different)
git remote add origin https://github.com/diwanaditya/cosmos-mas.git

# Push
git branch -M main
git push -u origin main
```

## Step 3 — Authenticate

When prompted for username/password:
- Username: diwanaditya
- Password: use a GitHub Personal Access Token (NOT your password)
  - Create one at: https://github.com/settings/tokens
  - Scopes needed: repo (full control)

## Step 4 — Add GitHub Actions CI (optional but recommended)

Create `.github/workflows/tests.yml` to auto-run tests on push:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: {python-version: '3.11'}
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
```

## Step 5 — Add to paper

The URL to cite in your paper is:
  https://github.com/diwanaditya/cosmos-mas

This is already written into the paper in §5.1, §10 Conclusion, and Appendix D.

## Verify

After pushing, confirm at:
  https://github.com/diwanaditya/cosmos-mas
