# Measuring Adaptive Interviewing 

Can we distinguish scripted interview questions from questions that are personalized or genuinely generated in response to a participant?

## Overview

This project analyzes the Anthropic Interviewer dataset to estimate how many interview questions change across three populations:

- Workforce = 1,000 interviews
- Creatives = 125 interviews 
- Scientists = 125 interviews

I built a three-layer classifier:

1. Verbatim / near-verbatim
2. Personalized spine
3. True branch

The goal was to build a reproducible measurement framework for for interviewer adaptation, not to seek perfect semantic ground truth.

## Results

| Population | Verbatim | Personalized | Branch |
|------------|----------|---------------|--------|
| Workforce  | 44.8%    | 16.5%         | 38.7%  |
| Creatives  | 70.4%    | 15.6%         | 14.0%  |
| Scientists | 46.7%    | 26.3%         | 27.0%  |

<img width="3496" height="1982" alt="01_question_architecture" src="https://github.com/user-attachments/assets/28ae0a34-e3be-4fed-a180-493e1d500f18" />

## The Important Finding: Measurement Can Lie

The first version of the pipeline reported extremely high branching:

| Population | Initial | Corrected |
|------------|---------|-----------|
| Workforce  | 82.0%   | 38.7%     |
| Creatives  | 88.3%   | 14.0%     |
| Scientists | 86.6%   | 27.0%     |

The difference was caused primarily by question extraction.

Compound interviewer turns such as:

> "Could you tell me about your workday?
> What tasks do you use it for?"

were being treated as multiple independent questions.

The corrected parser treats an assistant turn as one question item while preserving compound questions.


<img width="3496" height="1998" alt="02_branch_distribution" src="https://github.com/user-attachments/assets/e3c01ed2-9263-4a26-addb-c64bef134ca5" />


## Method

### 1. Transcript parsing

Assistant and User turns are aligned into question/answer pairs.

### 2. Question extraction

One assistant turn containing a question becomes one question item.

### 3. Semantic representation

Questions are embedded using:

`all-mpnet-base-v2`

### 4. Spine discovery

HDBSCAN identifies recurring question clusters.

Clusters reaching at least 20% of interviews are treated as candidate interview spine questions.

### 5. Template matching

Each question is compared against the nearest spine using:

- semantic similarity
- spine coverage
- question coverage
- sequence similarity

### 6. Classification

Questions are classified as:

**Verbatim**

Very high semantic similarity plus strong lexical/template preservation.

**Personalized**

Semantically related to a spine question while preserving enough of its underlying template.

**Branch**

Not sufficiently close to a discovered spine.

<img width="5271" height="1832" alt="03_semantic_template_map" src="https://github.com/user-attachments/assets/89d9b105-2d40-4e70-b30b-3754d9d8d772" />

## Validation

The pipeline was iteratively validated by inspecting borderline cases.

One important failure mode was that semantic similarity alone could identify questions that were topically related but functionally different.

For example, a question about *whether* a researcher experimented with AI can be semantically close to a question about *how* they use AI without being the same interview question.

This is why semantic similarity and template preservation are both used.

## Limitations

This is an exploratory measurement system, not a human-labeled ground-truth classifier.

Important limitations include:

- HDBSCAN-derived spines depend on clustering parameters.
- Semantic similarity does not equal identical question intent.
- Lexical/template metrics can miss heavily personalized questions.
- The three populations differ in sample size.
- "Branching" is operationally defined by the classifier.
- Results should be interpreted as estimates of interviewer behavior,
  not objective measurements of conversational intelligence.

## Reproducibility

[installation]

[run instructions]

[configuration]

## Project Structure

```text
.
├── analysis.py
├── README.md
├── requirements.txt
└── figures/
    ├── question_composition.png
    ├── parser_correction.png
    └── semantic_template_space.png



