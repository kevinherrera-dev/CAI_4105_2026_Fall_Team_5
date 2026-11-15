# AI Use Log: Challenge 2

The course syllabus allows AI with three principles:
1. AI cannot pass this course.
2. AI contributions must be attributed and true.
3. The use of AI tools should be open and documented.

This file is how our team documents it.

## What to log
**Log it** when AI did real work on the project in a session, for example:
- wrote or changed notebook code, the README, or other project files
- chose or suggested an approach we used (preprocessing, models, hyperparameters, final model)
- ran, checked or debugged the notebook

**No need to log** quick questions that didn't produce project work (e.g. "what does F1 mean?", "how do I rename a folder?").

**Also log manual work** when you did a meaningful part yourself (ran and checked results, reviewed a PR, wrote a section), so it's clear who did what.

## Rules
- One line per work session per person, newest at the bottom.
- Be honest and specific: name the tool, say what it produced, say what you checked yourself. Never mark AI-written code as manual.
- Only edit your own lines.

## Format
`- YYYY-MM-DD | Name | Tool (or "manual") | What AI did / what you did yourself`

## Log
- 2026-11-14 | Daniel Ibarra | Claude Code | Set up the Challenge 2 folder, AGENTS.md, AI_USE_LOG.md, README skeleton and the pull request template; explained the assignment, F1, precision and recall.
- 2026-11-14 | Daniel Ibarra | manual | Split the work across the team and chose the shared-file workflow.
- 2026-11-14 | Daniel Ibarra | Claude Code | Wrote and ran Piece 1 (EDA) of `step1-2_base.ipynb`: data checks, target balance, numeric ranges, category values, buy rates by group, and the summary table; updated AGENTS.md and README.md with the findings.
- 2026-11-14 | Daniel Ibarra | Claude Code | Wrote `helpers.py` (shared data loading, preprocessing, stratified 5-fold CV, `evaluate()`, results table, confusion matrix) and tested it with all 7 required models and GridSearchCV; wrote and ran Pieces 2-4 of `step1-2_base.ipynb` (preprocessing, validation, baselines); updated AGENTS.md and README.md.
- 2026-11-15 | Kevin Herrera | ChatGPT GPT-6 | Drafted `challenge2.ipynb` for Steps 5–6 (provisional model rationale, shared-helper CV, final fit and test prediction CSV assertions), pinned `requirements.txt`, and updated project documentation; notebook has not yet been executed or its predictions verified.
