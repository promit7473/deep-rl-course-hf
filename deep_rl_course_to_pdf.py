#!/usr/bin/env python3
"""
HuggingFace Deep RL Course → Single PDF
All 7 design upgrades:
  1. Full-bleed gradient cover page with course stats
  2. Per-unit color divider pages with giant faint unit number
  3. Running page headers (unit name + section title)
  4. Styled callout boxes (tip / warning / note)
  5. Section number badges on every section title
  6. Image drop-shadow + border treatment
  7. Per-unit accent color throughout
"""

import time, re, requests
from bs4 import BeautifulSoup
from weasyprint import HTML, CSS
import os

BASE_URL  = "https://huggingface.co/learn/deep-rl-course"
HF_ORIGIN = "https://huggingface.co"

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"}

# ── Per-unit accent color palette ────────────────────────────────────────────
UNIT_COLORS = [
    "#6366F1",  # Unit 0   – indigo
    "#0EA5E9",  # Unit 1   – sky
    "#10B981",  # Bonus 1  – emerald
    "#F59E0B",  # Live 1   – amber
    "#8B5CF6",  # Unit 2   – violet
    "#EF4444",  # Unit 3   – red
    "#14B8A6",  # Bonus 2  – teal
    "#F97316",  # Unit 4   – orange
    "#06B6D4",  # Unit 5   – cyan
    "#84CC16",  # Unit 6   – lime
    "#EC4899",  # Unit 7   – pink
    "#3B82F6",  # Unit 8-1 – blue
    "#7C3AED",  # Unit 8-2 – purple
    "#059669",  # Bonus 3  – green
    "#DC2626",  # Bonus 5  – crimson
    "#D97706",  # Cert     – gold
]

COURSE_PAGES = [
    ("Unit 0 — Welcome to the Course", "0", [
        ("unit0/introduction", "Welcome to the course"),
        ("unit0/setup",        "Setup"),
        ("unit0/discord101",   "Discord 101"),
    ]),
    ("Unit 1 — Introduction to Deep Reinforcement Learning", "1", [
        ("unit1/introduction",    "Introduction"),
        ("unit1/what-is-rl",      "What is Reinforcement Learning?"),
        ("unit1/rl-framework",    "The RL Framework"),
        ("unit1/tasks",           "The Type of Tasks"),
        ("unit1/exp-exp-tradeoff","The Exploration / Exploitation Tradeoff"),
        ("unit1/two-methods",     "Two Main Approaches for Solving RL"),
        ("unit1/deep-rl",         'The "Deep" in Deep RL'),
        ("unit1/summary",         "Summary"),
        ("unit1/glossary",        "Glossary"),
        ("unit1/hands-on",        "Hands-on"),
        ("unit1/quiz",            "Quiz"),
        ("unit1/conclusion",      "Conclusion"),
        ("unit1/additional-readings", "Additional Readings"),
    ]),
    ("Bonus Unit 1 — Deep RL with Huggy", "B1", [
        ("unitbonus1/introduction",    "Introduction"),
        ("unitbonus1/how-huggy-works", "How Huggy Works"),
        ("unitbonus1/train",           "Train Huggy"),
        ("unitbonus1/play",            "Play with Huggy"),
        ("unitbonus1/conclusion",      "Conclusion"),
    ]),
    ("Live 1 — How the Course Works, Q&A", "L1", [
        ("live1/live1", "Live 1 Session"),
    ]),
    ("Unit 2 — Introduction to Q-Learning", "2", [
        ("unit2/introduction",              "Introduction"),
        ("unit2/what-is-rl",                "What is RL? A Short Recap"),
        ("unit2/two-types-value-based-methods","Two Types of Value-Based Methods"),
        ("unit2/bellman-equation",          "The Bellman Equation"),
        ("unit2/mc-vs-td",                  "Monte Carlo vs Temporal Difference"),
        ("unit2/mid-way-recap",             "Mid-way Recap"),
        ("unit2/mid-way-quiz",              "Mid-way Quiz"),
        ("unit2/q-learning",                "Introducing Q-Learning"),
        ("unit2/q-learning-example",        "A Q-Learning Example"),
        ("unit2/q-learning-recap",          "Q-Learning Recap"),
        ("unit2/glossary",                  "Glossary"),
        ("unit2/hands-on",                  "Hands-on"),
        ("unit2/quiz2",                     "Q-Learning Quiz"),
        ("unit2/conclusion",                "Conclusion"),
        ("unit2/additional-readings",       "Additional Readings"),
    ]),
    ("Unit 3 — Deep Q-Learning with Atari Games", "3", [
        ("unit3/introduction",       "Introduction"),
        ("unit3/from-q-to-dqn",      "From Q-Learning to Deep Q-Learning"),
        ("unit3/deep-q-network",     "The Deep Q-Network (DQN)"),
        ("unit3/deep-q-algorithm",   "The Deep Q Algorithm"),
        ("unit3/glossary",           "Glossary"),
        ("unit3/hands-on",           "Hands-on"),
        ("unit3/quiz",               "Quiz"),
        ("unit3/conclusion",         "Conclusion"),
        ("unit3/additional-readings","Additional Readings"),
    ]),
    ("Bonus Unit 2 — Hyperparameter Tuning with Optuna", "B2", [
        ("unitbonus2/introduction", "Introduction"),
        ("unitbonus2/optuna",       "Optuna"),
        ("unitbonus2/hands-on",     "Hands-on"),
    ]),
    ("Unit 4 — Policy Gradient with PyTorch", "4", [
        ("unit4/introduction",            "Introduction"),
        ("unit4/what-are-policy-based-methods","Policy-Based Methods"),
        ("unit4/advantages-disadvantages","Advantages and Disadvantages"),
        ("unit4/policy-gradient",         "Diving Deeper into Policy-Gradient"),
        ("unit4/pg-theorem",              "(Optional) The Policy Gradient Theorem"),
        ("unit4/glossary",                "Glossary"),
        ("unit4/hands-on",                "Hands-on"),
        ("unit4/quiz",                    "Quiz"),
        ("unit4/conclusion",              "Conclusion"),
        ("unit4/additional-readings",     "Additional Readings"),
    ]),
    ("Unit 5 — Introduction to Unity ML-Agents", "5", [
        ("unit5/introduction",       "Introduction"),
        ("unit5/how-mlagents-works", "How ML-Agents Works"),
        ("unit5/snowball-target",    "The SnowballTarget Environment"),
        ("unit5/pyramids",           "The Pyramids Environment"),
        ("unit5/curiosity",          "(Optional) Curiosity in Deep RL"),
        ("unit5/hands-on",           "Hands-on"),
        ("unit5/bonus",              "Bonus: Create Your Own Environments"),
        ("unit5/quiz",               "Quiz"),
        ("unit5/conclusion",         "Conclusion"),
    ]),
    ("Unit 6 — Actor Critic Methods with Robotics", "6", [
        ("unit6/introduction",        "Introduction"),
        ("unit6/variance-problem",    "The Problem of Variance in Reinforce"),
        ("unit6/advantage-actor-critic","Advantage Actor Critic (A2C)"),
        ("unit6/hands-on",            "Hands-on: A2C with Panda-Gym"),
        ("unit6/quiz",                "Quiz"),
        ("unit6/conclusion",          "Conclusion"),
        ("unit6/additional-readings", "Additional Readings"),
    ]),
    ("Unit 7 — Multi-Agents and AI vs AI", "7", [
        ("unit7/introduction",         "Introduction"),
        ("unit7/introduction-to-marl", "Introduction to MARL"),
        ("unit7/multi-agent-setting",  "Designing Multi-Agent Systems"),
        ("unit7/self-play",            "Self-Play"),
        ("unit7/hands-on",             "Hands-on: Train Your Soccer Team"),
        ("unit7/quiz",                 "Quiz"),
        ("unit7/conclusion",           "Conclusion"),
        ("unit7/additional-readings",  "Additional Readings"),
    ]),
    ("Unit 8, Part 1 — Proximal Policy Optimization (PPO)", "8a", [
        ("unit8/introduction",              "Introduction"),
        ("unit8/intuition-behind-ppo",      "The Intuition Behind PPO"),
        ("unit8/clipped-surrogate-objective","The Clipped Surrogate Objective"),
        ("unit8/visualize",                 "Visualize the Clipped Surrogate"),
        ("unit8/hands-on-cleanrl",          "PPO with CleanRL"),
        ("unit8/conclusion",                "Conclusion"),
        ("unit8/additional-readings",       "Additional Readings"),
    ]),
    ("Unit 8, Part 2 — PPO with Doom", "8b", [
        ("unit8/introduction-sf", "Introduction"),
        ("unit8/hands-on-sf",     "PPO with Sample Factory and Doom"),
        ("unit8/conclusion-sf",   "Conclusion"),
    ]),
    ("Bonus Unit 3 — Advanced Topics in RL", "B3", [
        ("unitbonus3/introduction",       "Introduction"),
        ("unitbonus3/model-based",        "Model-Based Reinforcement Learning"),
        ("unitbonus3/offline-online",     "Offline vs. Online RL"),
        ("unitbonus3/generalisation",     "Generalisation in RL"),
        ("unitbonus3/rlhf",               "RL from Human Feedback (RLHF)"),
        ("unitbonus3/decision-transformers","Decision Transformers & Offline RL"),
        ("unitbonus3/language-models",    "Language Models in RL"),
        ("unitbonus3/curriculum-learning","(Auto) Curriculum Learning for RL"),
        ("unitbonus3/envs-to-try",        "Interesting Environments to Try"),
        ("unitbonus3/learning-agents",    "Unreal Learning Agents"),
        ("unitbonus3/godotrl",            "Godot RL"),
        ("unitbonus3/student-works",      "Students' Projects"),
        ("unitbonus3/rl-documentation",   "RL Documentation Intro"),
    ]),
    ("Bonus Unit 5 — Imitation Learning with Godot RL", "B5", [
        ("unitbonus5/introduction",          "Introduction"),
        ("unitbonus5/the-environment",       "The Environment"),
        ("unitbonus5/getting-started",       "Getting Started"),
        ("unitbonus5/train-our-robot",       "Train Our Robot"),
        ("unitbonus5/customize-the-environment","(Optional) Customize the Environment"),
        ("unitbonus5/conclusion",            "Conclusion"),
    ]),
    ("Certification and Congratulations", "🎓", [
        ("communication/conclusion",    "Congratulations"),
        ("communication/certification", "Get Your Certificate"),
    ]),
]

TOTAL_SECTIONS = sum(len(p) for _, _, p in COURSE_PAGES)

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS_BASE = """
/* ═══════════════════════════════════════════════════════
   PAGE SETUP
═══════════════════════════════════════════════════════ */
@page {
    size: A4;
    margin: 28mm 22mm 22mm 22mm;

    /* Running header bar */
    @top-left {
        content: string(unit-running);
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 7pt;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--accent, #6366F1);
        vertical-align: bottom;
        padding-bottom: 4px;
        border-bottom: 1.5px solid var(--accent, #6366F1);
    }
    @top-right {
        content: string(section-running);
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 7pt;
        color: #9CA3AF;
        vertical-align: bottom;
        padding-bottom: 4px;
        border-bottom: 1px solid #E5E7EB;
    }
    @bottom-right {
        content: counter(page);
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 8pt;
        color: #9CA3AF;
    }
    @bottom-left {
        content: "huggingface.co/learn/deep-rl-course";
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 7pt;
        color: #D1D5DB;
        letter-spacing: 0.03em;
    }
}

/* No running headers on cover / divider / toc pages */
@page cover-page   { margin: 0; }
@page cover-page   { @top-left { content: none; } @top-right { content: none; } @bottom-left { content: none; } @bottom-right { content: none; } }
@page divider-page { margin: 0; }
@page divider-page { @top-left { content: none; } @top-right { content: none; } @bottom-left { content: none; } @bottom-right { content: none; } }
@page toc-page     { @top-left { content: none; } @top-right { content: none; } }

/* ═══════════════════════════════════════════════════════
   BASE TYPOGRAPHY
═══════════════════════════════════════════════════════ */
body {
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 10.5pt;
    line-height: 1.78;
    color: #111827;
    background: white;
}

/* ═══════════════════════════════════════════════════════
   COVER PAGE
═══════════════════════════════════════════════════════ */
.cover {
    page: cover-page;
    page-break-after: always;
    width: 100%;
    height: 297mm;
    background: linear-gradient(145deg, #0F0F23 0%, #1a1a3e 45%, #0d1117 100%);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    padding: 0 28mm;
    box-sizing: border-box;
    position: relative;
}
.cover-bg-text {
    position: absolute;
    right: -10mm;
    top: 30mm;
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 180pt;
    font-weight: 900;
    color: rgba(255,255,255,0.03);
    line-height: 1;
    letter-spacing: -10px;
    pointer-events: none;
}
.cover-eyebrow {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 8pt;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #FF6B35;
    margin-bottom: 18px;
}
.cover-title {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 40pt;
    font-weight: 900;
    color: #FFFFFF;
    line-height: 1.1;
    margin-bottom: 6px;
    letter-spacing: -1px;
}
.cover-title-accent {
    color: #FF6B35;
}
.cover-subtitle {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 13pt;
    color: #9CA3AF;
    margin-bottom: 36px;
    font-weight: 300;
}
.cover-divider {
    width: 48px;
    height: 4px;
    background: #FF6B35;
    border-radius: 2px;
    margin-bottom: 36px;
    border: none;
}
.cover-stats {
    display: flex;
    gap: 0;
}
.cover-stat {
    margin-right: 28px;
}
.cover-stat-num {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 22pt;
    font-weight: 800;
    color: white;
    line-height: 1;
}
.cover-stat-label {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 7.5pt;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 3px;
}
.cover-footer {
    position: absolute;
    bottom: 20mm;
    left: 28mm;
    right: 28mm;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.cover-footer-url {
    font-family: 'Courier New', monospace;
    font-size: 8pt;
    color: #374151;
}
.cover-footer-badge {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 7.5pt;
    color: #4B5563;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 4px 10px;
    border-radius: 20px;
}

/* ═══════════════════════════════════════════════════════
   TABLE OF CONTENTS
═══════════════════════════════════════════════════════ */
.toc-wrap {
    page: toc-page;
    page-break-after: always;
}
.toc-heading {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 26pt;
    font-weight: 900;
    color: #111827;
    margin-bottom: 6px;
    letter-spacing: -0.5px;
}
.toc-heading-accent {
    display: block;
    width: 44px;
    height: 4px;
    background: #FF6B35;
    border-radius: 2px;
    margin-bottom: 24px;
    border: none;
}
.toc-unit {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 8.5pt;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 16px;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.toc-unit-badge {
    display: inline-block;
    font-size: 7pt;
    font-weight: 700;
    color: white;
    padding: 2px 7px;
    border-radius: 20px;
    line-height: 1.4;
    letter-spacing: 0.05em;
}
.toc-section {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 9pt;
    color: #6B7280;
    padding: 2px 0 2px 16px;
    border-left: 2px solid #F3F4F6;
    margin-left: 4px;
    margin-bottom: 1px;
    line-height: 1.5;
}

/* ═══════════════════════════════════════════════════════
   UNIT DIVIDER PAGE
═══════════════════════════════════════════════════════ */
.unit-divider {
    page: divider-page;
    page-break-before: always;
    page-break-after: always;
    width: 100%;
    height: 297mm;
    box-sizing: border-box;
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding: 20mm 22mm;
}
.unit-divider-bg {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 55%;
}
.unit-divider-watermark {
    position: absolute;
    right: -15mm;
    top: 18mm;
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 200pt;
    font-weight: 900;
    color: rgba(255,255,255,0.07);
    line-height: 1;
    letter-spacing: -8px;
}
.unit-divider-label {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 8pt;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 10px;
}
.unit-divider-title {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 30pt;
    font-weight: 900;
    color: #111827;
    line-height: 1.15;
    margin-bottom: 20px;
    max-width: 160mm;
    letter-spacing: -0.5px;
}
.unit-divider-rule {
    width: 44px;
    height: 4px;
    border-radius: 2px;
    border: none;
    margin-bottom: 16px;
}
.unit-divider-sections {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 8.5pt;
    color: #9CA3AF;
    line-height: 1.8;
}
.unit-divider-section-dot {
    display: inline-block;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}

/* ═══════════════════════════════════════════════════════
   RUNNING STRING MARKERS (invisible)
═══════════════════════════════════════════════════════ */
.unit-marker {
    string-set: unit-running content();
    display: block;
    height: 0;
    overflow: hidden;
    font-size: 0;
    color: transparent;
}
.section-marker {
    string-set: section-running content();
    display: block;
    height: 0;
    overflow: hidden;
    font-size: 0;
    color: transparent;
}

/* ═══════════════════════════════════════════════════════
   SECTION HEADER (within a unit)
═══════════════════════════════════════════════════════ */
.section-header {
    margin-top: 10mm;
    margin-bottom: 7mm;
    page-break-after: avoid;
}
.section-header-row {
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-badge {
    display: inline-block;
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 7.5pt;
    font-weight: 800;
    color: white;
    padding: 3px 9px;
    border-radius: 20px;
    line-height: 1.4;
    letter-spacing: 0.05em;
    white-space: nowrap;
    flex-shrink: 0;
}
.section-title {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 14pt;
    font-weight: 800;
    color: #111827;
    line-height: 1.2;
}
.section-rule {
    height: 1px;
    background: #F3F4F6;
    border: none;
    margin-top: 8px;
    margin-bottom: 0;
}

/* ═══════════════════════════════════════════════════════
   CONTENT HEADINGS
═══════════════════════════════════════════════════════ */
h1 {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 19pt;
    font-weight: 900;
    color: #111827;
    margin-top: 10mm;
    margin-bottom: 4mm;
    line-height: 1.2;
    letter-spacing: -0.3px;
    page-break-after: avoid;
}
h2 {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 13.5pt;
    font-weight: 800;
    color: #1F2937;
    margin-top: 9mm;
    margin-bottom: 3mm;
    padding-bottom: 5px;
    border-bottom: 1.5px solid #F3F4F6;
    page-break-after: avoid;
}
h3 {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 11pt;
    font-weight: 700;
    margin-top: 6mm;
    margin-bottom: 2mm;
    page-break-after: avoid;
    /* color set inline per unit */
}
h4 {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 9.5pt;
    font-weight: 700;
    color: #6B7280;
    margin-top: 5mm;
    margin-bottom: 1.5mm;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    page-break-after: avoid;
}
h5, h6 {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 9.5pt;
    font-weight: 700;
    color: #374151;
    margin-top: 4mm;
    margin-bottom: 1mm;
    page-break-after: avoid;
}

/* ═══════════════════════════════════════════════════════
   BODY TEXT
═══════════════════════════════════════════════════════ */
p {
    margin-top: 0;
    margin-bottom: 8px;
    orphans: 3;
    widows: 3;
    text-align: justify;
    hyphens: auto;
    -webkit-hyphens: auto;
}
li {
    text-align: justify;
    hyphens: auto;
}
a { color: #FF6B35; text-decoration: none; }
strong { font-weight: 700; color: #111827; }
em { font-style: italic; color: #374151; }

/* ═══════════════════════════════════════════════════════
   LISTS
═══════════════════════════════════════════════════════ */
ul, ol { margin: 5px 0 10px 0; padding-left: 22px; }
li { margin-bottom: 4px; line-height: 1.65; }
li > ul, li > ol { margin-top: 3px; margin-bottom: 3px; }

/* ═══════════════════════════════════════════════════════
   CODE
═══════════════════════════════════════════════════════ */
code {
    font-family: 'Courier New', Courier, monospace;
    font-size: 8.5pt;
    background: #F3F4F6;
    color: #B91C1C;
    padding: 1px 5px;
    border-radius: 4px;
    border: 1px solid #E5E7EB;
}
pre {
    background: #13131f;
    color: #e2e8f0;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0 14px 0;
    font-family: 'Courier New', Courier, monospace;
    font-size: 8.5pt;
    line-height: 1.65;
    white-space: pre-wrap;
    word-wrap: break-word;
    overflow-wrap: break-word;
    page-break-inside: avoid;
    /* border-left color set per unit inline */
    border-left: 4px solid #FF6B35;
}
pre code {
    background: none;
    border: none;
    padding: 0;
    color: inherit;
    font-size: inherit;
}

/* ═══════════════════════════════════════════════════════
   CALLOUT BOXES
═══════════════════════════════════════════════════════ */
blockquote {
    border-left: 4px solid #E5E7EB;
    margin: 10px 0 14px 0;
    padding: 10px 16px;
    background: #F9FAFB;
    color: #374151;
    border-radius: 0 6px 6px 0;
    page-break-inside: avoid;
}
blockquote p { margin: 0; }

/* Tip callout */
blockquote.tip {
    background: #FFFBEB;
    border-left-color: #F59E0B;
    border-radius: 0 8px 8px 0;
}
blockquote.tip::before {
    content: "💡  Tip";
    display: block;
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 7.5pt;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #B45309;
    margin-bottom: 5px;
}
blockquote.warning {
    background: #FEF2F2;
    border-left-color: #EF4444;
}
blockquote.warning::before {
    content: "⚠️  Warning";
    display: block;
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 7.5pt;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #991B1B;
    margin-bottom: 5px;
}
blockquote.note {
    background: #EFF6FF;
    border-left-color: #3B82F6;
}
blockquote.note::before {
    content: "📝  Note";
    display: block;
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 7.5pt;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #1D4ED8;
    margin-bottom: 5px;
}

/* ═══════════════════════════════════════════════════════
   IMAGES
═══════════════════════════════════════════════════════ */
img {
    max-width: 90%;
    height: auto;
    display: block;
    margin: 14px auto 18px auto;
    border-radius: 8px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 4px 16px rgba(0,0,0,0.10), 0 1px 4px rgba(0,0,0,0.06);
    page-break-inside: avoid;
}

/* ═══════════════════════════════════════════════════════
   TABLES
═══════════════════════════════════════════════════════ */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0 16px 0;
    font-size: 9pt;
    font-family: 'Helvetica Neue', Arial, sans-serif;
    page-break-inside: avoid;
}
thead tr { color: white; }
thead th {
    padding: 8px 12px;
    text-align: left;
    font-weight: 700;
    border: none;
}
tbody tr:nth-child(even) { background: #F9FAFB; }
tbody td {
    padding: 7px 12px;
    border-bottom: 1px solid #E5E7EB;
    vertical-align: top;
}

/* ═══════════════════════════════════════════════════════
   UTILITY
═══════════════════════════════════════════════════════ */
.page-break { page-break-after: always; break-after: page; }
.no-break   { page-break-inside: avoid; break-inside: avoid; }

/* Strip HuggingFace UI chrome */
nav, aside, footer, dialog, button, form,
[class*="sticky"], [class*="SVELTE"],
[class*="DocNotebook"], [class*="EditPage"],
[class*="FrameworkContent"], [class*="header-link"],
[class*="bg-linear-to-br"] { display: none !important; }
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

JUNK_CLASSES = [
    "bg-linear-to-br",   # HF sign-up promo
    "SVELTE_HYDRATER",
    "DocNotebook",
    "EditPage",
    "FrameworkContent",
    "header-link",
    "sticky",
]

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

def fetch_page_content(path: str) -> str:
    url = f"{BASE_URL}/{path}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=25)
        resp.raise_for_status()
    except Exception as e:
        return f"<p><em>Could not load: {url} — {e}</em></p>"

    soup = BeautifulSoup(resp.text, "lxml")
    content = soup.select_one("div.prose-doc")
    if not content:
        for sel in ["div.prose", "article", "main"]:
            content = soup.select_one(sel)
            if content: break
    if not content:
        content = soup.body or soup

    # Strip junk tags
    for tag in content.find_all(["nav","header","footer","script","style",
                                  "button","aside","dialog","iframe","form","svg"]):
        tag.decompose()

    for tag in list(content.find_all(True)):
        if tag.parent is None: continue
        if not hasattr(tag, "get"): continue
        cls = " ".join(tag.get("class", []))
        if any(j in cls for j in JUNK_CLASSES):
            tag.decompose()

    # Fix relative image srcs
    for img in content.find_all("img"):
        src = img.get("src", "")
        if src.startswith("/"):
            img["src"] = HF_ORIGIN + src

    return str(content)


def build_cover() -> str:
    return f"""
<div class="cover">
  <div class="cover-bg-text">RL</div>
  <div class="cover-eyebrow">Hugging Face · Full Course Compilation</div>
  <div class="cover-title">
    Deep<br/><span class="cover-title-accent">Reinforcement</span><br/>Learning
  </div>
  <div class="cover-subtitle">From Beginner to Expert</div>
  <hr class="cover-divider"/>
  <div class="cover-stats">
    <div class="cover-stat">
      <div class="cover-stat-num">16</div>
      <div class="cover-stat-label">Units</div>
    </div>
    <div class="cover-stat">
      <div class="cover-stat-num">{TOTAL_SECTIONS}</div>
      <div class="cover-stat-label">Sections</div>
    </div>
    <div class="cover-stat">
      <div class="cover-stat-num">∞</div>
      <div class="cover-stat-label">Things to Learn</div>
    </div>
  </div>
  <div class="cover-footer">
    <div class="cover-footer-url">huggingface.co/learn/deep-rl-course</div>
    <div class="cover-footer-badge">🤗 Open Source &amp; Free</div>
  </div>
</div>
"""


def build_toc() -> str:
    html = '<div class="toc-wrap">\n'
    html += '<div class="toc-heading">Contents</div>\n'
    html += '<hr class="toc-heading-accent"/>\n'
    for idx, (unit_name, unit_num, pages) in enumerate(COURSE_PAGES):
        color = UNIT_COLORS[idx % len(UNIT_COLORS)]
        badge = f'<span class="toc-unit-badge" style="background:{color};">{unit_num}</span>'
        html += f'<div class="toc-unit">{badge} {unit_name}</div>\n'
        for _, title in pages:
            html += f'<div class="toc-section">› {title}</div>\n'
    html += '</div>\n'
    return html


def build_unit_divider(unit_name: str, unit_num: str, pages: list, color: str) -> str:
    light_color = hex_to_rgba(color, 0.08)
    section_lines = "\n".join(
        f'<div><span class="unit-divider-section-dot" style="background:{color};"></span>{t}</div>'
        for _, t in pages[:6]
    )
    if len(pages) > 6:
        section_lines += f'\n<div style="color:#D1D5DB;">+ {len(pages)-6} more sections</div>'

    return f"""
<div class="unit-divider" style="background: linear-gradient(160deg, {hex_to_rgba(color,0.12)} 0%, #ffffff 55%);">
  <div class="unit-divider-watermark" style="color:rgba(0,0,0,0.04);">{unit_num}</div>
  <div class="unit-divider-label">Chapter {unit_num}</div>
  <div class="unit-divider-title">{unit_name}</div>
  <hr class="unit-divider-rule" style="background:{color};"/>
  <div class="unit-divider-sections">
    {section_lines}
  </div>
</div>
"""


def build_section_header(title: str, badge_label: str, color: str) -> str:
    return f"""
<div class="section-header no-break">
  <div class="section-header-row">
    <span class="section-badge" style="background:{color};">{badge_label}</span>
    <span class="section-title">{title}</span>
  </div>
  <hr class="section-rule"/>
</div>
<span class="section-marker">{title}</span>
"""


def build_pdf():
    print("🚀  Deep RL Course → PDF  (upgraded)\n")

    parts = []
    parts.append(build_cover())
    parts.append(build_toc())
    parts.append('<div class="page-break"></div>')

    fetched = 0

    for unit_idx, (unit_name, unit_num, pages) in enumerate(COURSE_PAGES):
        color = UNIT_COLORS[unit_idx % len(UNIT_COLORS)]
        print(f"\n{'─'*60}")
        print(f"📚  {unit_name}  [{color}]")

        parts.append(build_unit_divider(unit_name, unit_num, pages, color))

        # Running unit header marker (invisible, picked up by @top-left)
        parts.append(f'<span class="unit-marker">{unit_name}</span>')

        for sec_idx, (path, title) in enumerate(pages):
            fetched += 1
            badge_label = f"{unit_num}.{sec_idx+1}"
            print(f"  [{fetched:>3}/{TOTAL_SECTIONS}]  {badge_label}  {title}...", end=" ", flush=True)

            content = fetch_page_content(path)

            # Inject per-unit color on h3 and pre via inline style attribute surgery
            # (WeasyPrint doesn't support CSS custom properties from Python-injected HTML)
            content = content.replace("<h3 ", f'<h3 style="color:{color};" ')
            content = content.replace("<pre>", f'<pre style="border-left-color:{color};">')
            content = content.replace("<pre ", f'<pre style="border-left-color:{color};" ')
            content = content.replace("<thead>", f'<thead style="background:{color};">')

            section = build_section_header(title, badge_label, color)
            section += content

            if sec_idx < len(pages) - 1:
                section += '<div class="page-break"></div>'

            parts.append(section)
            print("✓")
            time.sleep(0.4)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>HuggingFace Deep Reinforcement Learning Course</title>
</head>
<body>
{''.join(parts)}
</body>
</html>"""

    output = os.path.expanduser("~/deep_rl_course.pdf")
    print(f"\n\n📄  Rendering PDF — please wait...\n")
    HTML(string=full_html, base_url=HF_ORIGIN).write_pdf(
        output,
        stylesheets=[CSS(string=CSS_BASE)],
        optimize_images=True,
        jpeg_quality=82,
        uncompressed_pdf=False,
    )
    mb = os.path.getsize(output) / 1024 / 1024
    print(f"✅  Saved → {output}  ({mb:.1f} MB)")


if __name__ == "__main__":
    build_pdf()
