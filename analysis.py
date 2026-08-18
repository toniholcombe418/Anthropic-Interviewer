from datasets import load_dataset
import re
import collections
from difflib import SequenceMatcher

import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "all-mpnet-base-v2"

SIM_VERBATIM = 0.90
SIM_SPINE = 0.60

REACH_MIN = 0.20
MERGE_COS = 0.80

VERBATIM_SPINE_COVERAGE = 0.80
VERBATIM_Q_COVERAGE = 0.80
VERBATIM_SEQUENCE_SIM = 0.78

PERSONALIZED_SPINE_COVERAGE = 0.45
PERSONALIZED_SEQUENCE_SIM = 0.40

SKIP_FIRST_ASSISTANT_TURN = True


# ============================================================
# STOPWORDS
# ============================================================

STOP = set(
    """
    a an the and or but if then of to in on at for with from by as
    is are was were be been being this that these those it its i
    you he she we they what how when where why do does did have has
    had will would could should can may might your my our their his
    her me not so just really
    """.split()
)


# ============================================================
# TEXT / TOKEN HELPERS
# ============================================================

def normalize_text(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def content_tokens(s):
    return [
        t
        for t in re.findall(r"[a-z]+", s.lower())
        if t not in STOP and len(t) > 2
    ]


def content_words(s):
    return set(content_tokens(s))


def sequence_similarity(a_tokens, b_tokens):
    if not a_tokens or not b_tokens:
        return 0.0

    return SequenceMatcher(
        None,
        a_tokens,
        b_tokens,
        autojunk=False,
    ).ratio()


def lexical_template_metrics(question, spine_question):

    q_tokens = content_tokens(question)
    s_tokens = content_tokens(spine_question)

    q_words = set(q_tokens)
    s_words = set(s_tokens)

    shared = q_words & s_words

    spine_coverage = (
        len(shared) / len(s_words)
        if s_words
        else 0.0
    )

    q_coverage = (
        len(shared) / len(q_words)
        if q_words
        else 0.0
    )

    seq_sim = sequence_similarity(
        q_tokens,
        s_tokens,
    )

    return {
        "spine_coverage": spine_coverage,
        "q_coverage": q_coverage,
        "sequence_sim": seq_sim,
        "shared_words": shared,
    }


# ============================================================
# TRANSCRIPT PARSING
# ============================================================

def parse_turns(text):
    """
    Parse transcript into:

        [(speaker, body), ...]

    Supports Assistant, AI, and User labels.
    """

    parts = re.split(
        r"\n(?=(?:Assistant|AI|User):\s*)",
        text,
    )

    turns = []

    for p in (x.strip() for x in parts):

        if not p or ":" not in p:
            continue

        speaker, body = p.split(":", 1)

        speaker = speaker.strip()
        body = body.strip()

        if speaker in ("Assistant", "AI", "User"):
            turns.append(
                (speaker, body)
            )

    return turns


# ============================================================
# QUESTION EXTRACTION
# ============================================================

def extract_questions(assistant_turn):
    """
    Return exactly ONE question item per assistant turn.

    Rules:
      - If there is no '?' -> return [].
      - Never split compound questions.
      - Keep all questions in the same assistant turn together.
      - Remove obvious introductory/reflection prose.
      - Remove obvious closing/interview-wrap-up prose.
    """

    if not assistant_turn:
        return []

    text = str(assistant_turn).strip()

    # Must contain an actual question mark.
    if "?" not in text:
        return []

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    # --------------------------------------------------------
    # Find the first question.
    #
    # Keep the sentence containing the first '?' and everything
    # after it, so compound questions stay together.
    # --------------------------------------------------------

    first_q = text.find("?")

    sentence_start = 0

    for m in re.finditer(
        r"[.!?]\s+",
        text[:first_q],
    ):
        sentence_start = m.end()

    question_text = text[sentence_start:].strip()

    # --------------------------------------------------------
    # Remove closing boilerplate.
    # --------------------------------------------------------

    closing_patterns = [
        r"\bThose are all the questions I had prepared\b",
        r"\bBefore we wrap up\b",
        r"\bBefore I let you go\b",
        r"\bThat covers everything\b",
        r"\bThat's all the questions\b",
    ]

    cut_positions = []

    for pattern in closing_patterns:

        m = re.search(
            pattern,
            question_text,
            flags=re.IGNORECASE,
        )

        if m:
            cut_positions.append(
                m.start()
            )

    if cut_positions:

        question_text = question_text[
            :min(cut_positions)
        ].strip()

    # --------------------------------------------------------
    # Conservative opener cleanup.
    #
    # This catches cases where reflection and the question are
    # accidentally part of the same sentence.
    # --------------------------------------------------------

    opener_pattern = re.compile(
        r"\b(?:"
        r"Could you\b|"
        r"Can you\b|"
        r"Would you\b|"
        r"How do you\b|"
        r"How have you\b|"
        r"How did you\b|"
        r"How does\b|"
        r"How would you\b|"
        r"What do you\b|"
        r"What did you\b|"
        r"What does\b|"
        r"What is\b|"
        r"What was\b|"
        r"What makes\b|"
        r"Where do you\b|"
        r"Where in\b|"
        r"When do you\b|"
        r"When did you\b|"
        r"Why do you\b|"
        r"Why did you\b|"
        r"Tell me\b|"
        r"Walk me through\b|"
        r"Describe\b|"
        r"Looking ahead\b|"
        r"Are there\b|"
        r"Are you\b|"
        r"Is there\b|"
        r"Is this\b|"
        r"Have you\b|"
        r"Do you\b"
        r")",
        flags=re.IGNORECASE,
    )

    match = opener_pattern.search(
        question_text
    )

    if match and match.start() > 0:

        prefix = question_text[
            :match.start()
        ].strip()

        if prefix and not prefix.endswith(":"):

            question_text = question_text[
                match.start():
            ].strip()

    # Final cleanup.
    question_text = re.sub(
        r"\s+",
        " ",
        question_text,
    ).strip()

    # Safety check.
    if "?" not in question_text:
        return []

    return [question_text]


# ============================================================
# QUESTION / ANSWER PAIRS
# ============================================================

def qa_pairs(
    text,
    skip_first_assistant_turn=True,
):
    """
    Return:

        [(assistant_question, following_user_answer), ...]

    IMPORTANT:

    Each Assistant turn is ONE question item.

    Compound turns remain intact.

    The User turn immediately following that Assistant turn
    becomes its answer.
    """

    turns = parse_turns(text)

    pairs = []

    assistant_turn_index = 0
    pending_question = None

    for speaker, body in turns:

        if speaker in ("Assistant", "AI"):

            assistant_turn_index += 1

            # Skip opening assistant turn.
            if (
                skip_first_assistant_turn
                and assistant_turn_index == 1
            ):
                pending_question = None
                continue

            # ONE assistant turn = ONE question item.
            extracted = extract_questions(body)

            if extracted:
                pending_question = extracted[0]
            else:
                pending_question = None

        elif speaker == "User":

            if pending_question is not None:

                pairs.append(
                    (
                        pending_question,
                        body,
                    )
                )

                pending_question = None

    return pairs


# ============================================================
# UNION-FIND
# ============================================================

def union_find_groups(
    labels,
    emb,
    merge_cos,
):

    labs = sorted(
        int(l)
        for l in set(labels)
        if l != -1
    )

    if not labs:

        return (
            {},
            np.full(
                len(labels),
                -2,
                dtype=int,
            ),
        )

    cents = np.array([
        emb[labels == l].mean(axis=0)
        for l in labs
    ])

    norms = np.linalg.norm(
        cents,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        1e-12,
    )

    cents = cents / norms

    sim = cents @ cents.T

    parent = list(
        range(len(labs))
    )

    def find(a):

        while parent[a] != a:

            parent[a] = parent[
                parent[a]
            ]

            a = parent[a]

        return a

    def union(a, b):

        ra = find(a)
        rb = find(b)

        if ra != rb:
            parent[rb] = ra

    for a in range(len(labs)):

        for b in range(a + 1, len(labs)):

            if sim[a, b] >= merge_cos:
                union(a, b)

    roots = {}
    next_group = 0
    grp = {}

    for k, lab in enumerate(labs):

        root = find(k)

        if root not in roots:

            roots[root] = next_group
            next_group += 1

        grp[lab] = roots[root]

    member = np.array([
        grp[int(l)]
        if l != -1
        else -2
        for l in labels
    ])

    return grp, member


# ============================================================
# TEMPLATE MATCHING
# ============================================================

def nearest_spine_template(
    question,
    spine_questions,
    spine_cents,
    question_embedding,
):

    if len(spine_questions) == 0:

        return {
            "sim": 0.0,
            "spine_question": "",
            "spine_coverage": 0.0,
            "q_coverage": 0.0,
            "sequence_sim": 0.0,
            "shared_words": set(),
        }

    sims = (
        spine_cents
        @ question_embedding
    )

    best_idx = int(
        np.argmax(sims)
    )

    best_sim = float(
        sims[best_idx]
    )

    best_spine_q = (
        spine_questions[best_idx]
    )

    metrics = lexical_template_metrics(
        question,
        best_spine_q,
    )

    return {
        "sim": best_sim,
        "spine_question": best_spine_q,
        **metrics,
    }


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_question(
    sim,
    spine_coverage,
    q_coverage,
    sequence_sim,
):

    # Verbatim / near-verbatim spine.
    if (
        sim >= SIM_VERBATIM
        and spine_coverage >= VERBATIM_SPINE_COVERAGE
        and q_coverage >= VERBATIM_Q_COVERAGE
        and sequence_sim >= VERBATIM_SEQUENCE_SIM
    ):
        return "verbatim"

    # Personalized spine.
    if (
        sim >= SIM_SPINE
        and (
            spine_coverage >= PERSONALIZED_SPINE_COVERAGE
            or sequence_sim >= PERSONALIZED_SEQUENCE_SIM
        )
    ):
        return "personalized"

    # True branch.
    return "branch"


# ============================================================
# PER-SPLIT ANALYSIS
# ============================================================

def analyze_split(
    name,
    dataset,
    model,
):

    q_list = []
    a_list = []
    t_list = []

    for i, row in enumerate(dataset):

        pairs = qa_pairs(
            row["text"],
            skip_first_assistant_turn=(
                SKIP_FIRST_ASSISTANT_TURN
            ),
        )

        for q, a in pairs:

            q_list.append(q)
            a_list.append(a)
            t_list.append(i)

    n_int = len(dataset)
    n_q = len(q_list)

    if n_q == 0:

        return {
            "name": name,
            "n_int": n_int,
            "n_q": 0,
            "spine": [],
            "best_sim": np.array([]),
            "echo": np.array([]),
            "spine_coverage": np.array([]),
            "q_coverage": np.array([]),
            "sequence_sim": np.array([]),
            "lay": np.array([], dtype=object),
            "t_idx": np.array([], dtype=int),
            "pshare": np.array([]),
            "pershare": np.array([]),
            "q": [],
            "a": [],
            "nearest_spine": [],
            "shared_words": [],
        }

    t_idx = np.array(
        t_list,
        dtype=int,
    )

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    emb = model.encode(
        q_list,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    emb = np.asarray(emb)

    # --------------------------------------------------------
    # HDBSCAN
    # --------------------------------------------------------

    mcs = max(
        15,
        n_int // 20,
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        metric="euclidean",
    )

    labels = clusterer.fit_predict(
        emb
    )

    # --------------------------------------------------------
    # MERGE CLUSTERS
    # --------------------------------------------------------

    grp, member = union_find_groups(
        labels,
        emb,
        MERGE_COS,
    )

    # --------------------------------------------------------
    # REACH
    # --------------------------------------------------------

    reach = collections.defaultdict(set)

    for g, ti in zip(
        member,
        t_idx,
    ):

        if g != -2:

            reach[int(g)].add(
                int(ti)
            )

    if n_int > 0:

        spine_groups = sorted(
            (
                g
                for g in reach
                if (
                    len(reach[g])
                    / n_int
                ) >= REACH_MIN
            ),
            key=lambda G: -len(
                reach[G]
            ),
        )

    else:

        spine_groups = []

    # --------------------------------------------------------
    # SPINE CENTROIDS
    # --------------------------------------------------------

    spine_cents = []
    spine_rows = []

    for g in spine_groups:

        idx = np.where(
            member == g
        )[0]

        c = emb[idx].mean(
            axis=0
        )

        norm = np.linalg.norm(c)

        if norm > 0:
            c = c / norm

        spine_cents.append(c)

        best_local = int(
            np.argmax(
                emb[idx] @ c
            )
        )

        best = idx[best_local]

        spine_rows.append({
            "group": int(g),
            "reach": len(
                reach[g]
            ),
            "question": q_list[best],
        })

    if spine_cents:

        spine_cents = np.asarray(
            spine_cents
        )

    else:

        spine_cents = np.zeros(
            (
                0,
                emb.shape[1],
            )
        )

    spine_questions = [
        row["question"]
        for row in spine_rows
    ]

    # --------------------------------------------------------
    # NEAREST SPINE
    # --------------------------------------------------------

    best_sim = np.zeros(n_q)
    spine_coverage = np.zeros(n_q)
    q_coverage = np.zeros(n_q)
    sequence_sim = np.zeros(n_q)

    nearest_spine = []
    shared_words = []

    for k in range(n_q):

        result = nearest_spine_template(
            q_list[k],
            spine_questions,
            spine_cents,
            emb[k],
        )

        best_sim[k] = result["sim"]

        spine_coverage[k] = (
            result["spine_coverage"]
        )

        q_coverage[k] = (
            result["q_coverage"]
        )

        sequence_sim[k] = (
            result["sequence_sim"]
        )

        nearest_spine.append(
            result["spine_question"]
        )

        shared_words.append(
            result["shared_words"]
        )

    # --------------------------------------------------------
    # ANSWER OVERLAP
    # --------------------------------------------------------

    echo = np.array([
        (
            len(
                content_words(q)
                & content_words(a)
            )
            / len(
                content_words(q)
            )
        )
        if content_words(q)
        else 0.0
        for q, a in zip(
            q_list,
            a_list,
        )
    ])

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    lay = np.empty(
        n_q,
        dtype=object,
    )

    for k in range(n_q):

        lay[k] = classify_question(
            sim=best_sim[k],
            spine_coverage=(
                spine_coverage[k]
            ),
            q_coverage=q_coverage[k],
            sequence_sim=(
                sequence_sim[k]
            ),
        )

    # --------------------------------------------------------
    # PER-INTERVIEW BRANCH SHARE
    # --------------------------------------------------------

    total = collections.Counter(
        t_idx.tolist()
    )

    branch_count = collections.Counter(
        t_idx[
            lay == "branch"
        ].tolist()
    )

    pshare = np.array([
        branch_count.get(t, 0)
        / total[t]
        for t in range(n_int)
        if total[t] > 0
    ])

    # --------------------------------------------------------
    # PER-INTERVIEW PERSONALIZED SHARE
    # --------------------------------------------------------

    personalized_count = collections.Counter(
        t_idx[
            lay == "personalized"
        ].tolist()
    )

    pershare = np.array([
        personalized_count.get(t, 0)
        / total[t]
        for t in range(n_int)
        if total[t] > 0
    ])

    return {
        "name": name,
        "n_int": n_int,
        "n_q": n_q,

        "spine": spine_rows,

        "best_sim": best_sim,
        "echo": echo,

        "spine_coverage": spine_coverage,
        "q_coverage": q_coverage,
        "sequence_sim": sequence_sim,

        "lay": lay,
        "t_idx": t_idx,

        "pshare": pshare,
        "pershare": pershare,

        "q": q_list,
        "a": a_list,

        "nearest_spine": nearest_spine,
        "shared_words": shared_words,
    }


# ============================================================
# REPORT
# ============================================================

def safe_percentile(x, p):

    if len(x) == 0:
        return np.nan

    return float(
        np.percentile(x, p)
    )


def report(res):

    n = res["n_q"]

    print(
        "\n================  "
        f"{res['name'].upper()}  "
        "================"
    )

    print(
        f"interviews {res['n_int']}   "
        f"questions {n}"
    )

    if n == 0:

        print(
            "No questions found."
        )

        return

    # --------------------------------------------------------
    # SPINE
    # --------------------------------------------------------

    print(
        "\nspine "
        "(reach = # interviews asked):"
    )

    for row in res["spine"]:

        print(
            f"  {row['reach']:4d}  "
            f"{row['question'][:90]}"
        )

    if not res["spine"]:

        print(
            "  (none found — "
            "check thresholds for this split)"
        )

    # --------------------------------------------------------
    # SEMANTIC SIMILARITY
    # --------------------------------------------------------

    bs = res["best_sim"]

    print(
        "\nsemantic similarity percentiles:"
    )

    print(
        "  "
        + "  ".join(
            [
                f"p{p}:{safe_percentile(bs, p):.2f}"
                for p in (
                    10,
                    25,
                    50,
                    75,
                    90,
                )
            ]
        )
    )

    print(
        f"  >= {SIM_SPINE:.2f}: "
        f"{(bs >= SIM_SPINE).mean():.1%}"
    )

    print(
        f"  >= {SIM_VERBATIM:.2f}: "
        f"{(bs >= SIM_VERBATIM).mean():.1%}"
    )

    # --------------------------------------------------------
    # TEMPLATE METRICS
    # --------------------------------------------------------

    sc = res["spine_coverage"]
    qc = res["q_coverage"]
    ss = res["sequence_sim"]

    print(
        "\nspine-template metrics:"
    )

    print(
        "  spine coverage "
        + "  ".join(
            [
                f"p{p}:{safe_percentile(sc, p):.2f}"
                for p in (
                    10,
                    25,
                    50,
                    75,
                    90,
                )
            ]
        )
    )

    print(
        "  question coverage "
        + "  ".join(
            [
                f"p{p}:{safe_percentile(qc, p):.2f}"
                for p in (
                    10,
                    25,
                    50,
                    75,
                    90,
                )
            ]
        )
    )

    print(
        "  sequence similarity "
        + "  ".join(
            [
                f"p{p}:{safe_percentile(ss, p):.2f}"
                for p in (
                    10,
                    25,
                    50,
                    75,
                    90,
                )
            ]
        )
    )

    # --------------------------------------------------------
    # THREE-LAYER SPLIT
    # --------------------------------------------------------

    c = collections.Counter(
        res["lay"]
    )

    print(
        "\nthree-layer split:"
    )

    for k in (
        "verbatim",
        "personalized",
        "branch",
    ):

        print(
            f"  {k:13s} "
            f"{c[k]:5d}  "
            f"{c[k] / n:5.1%}"
        )

    scripted = (
        c["verbatim"]
        + c["personalized"]
    ) / n

    print(
        f"  -> scripted spine "
        f"{scripted:.1%}"
    )

    print(
        f"  -> true branch "
        f"{c['branch'] / n:.1%}"
    )

    # --------------------------------------------------------
    # NON-VERBATIM COMPOSITION
    # --------------------------------------------------------

    print(
        "\ncomposition of non-verbatim questions:"
    )

    print(
        f"  personalized spine: "
        f"{c['personalized'] / n:.1%} of all"
    )

    print(
        f"  true branch: "
        f"{c['branch'] / n:.1%} of all"
    )

    nonverbatim = (
        c["personalized"]
        + c["branch"]
    )

    if nonverbatim > 0:

        branch_among_nonverbatim = (
            c["branch"]
            / nonverbatim
        )

    else:

        branch_among_nonverbatim = np.nan

    print(
        f"  true branch among non-verbatim: "
        f"{branch_among_nonverbatim:.1%}"
    )

    # --------------------------------------------------------
    # PER-INTERVIEW BRANCH SHARE
    # --------------------------------------------------------

    ps = res["pshare"]

    print(
        "\nper-interview TRUE branch share:"
    )

    if len(ps):

        quartiles = np.quantile(
            ps,
            [
                0.25,
                0.50,
                0.75,
            ],
        )

        print(
            f"  mean {ps.mean():.3f}  "
            f"median {np.median(ps):.3f}  "
            f"quartiles "
            f"{np.round(quartiles, 3)}"
        )

        print(
            f"  under 10% branch: "
            f"{(ps < 0.10).mean():.1%}"
        )

        print(
            f"  over 40% branch: "
            f"{(ps > 0.40).mean():.1%}"
        )

    else:

        print(
            "  no interviews with questions"
        )

    # --------------------------------------------------------
    # PER-INTERVIEW PERSONALIZED SHARE
    # --------------------------------------------------------

    pers = res["pershare"]

    print(
        "\nper-interview PERSONALIZED-SPINE share:"
    )

    if len(pers):

        quartiles = np.quantile(
            pers,
            [
                0.25,
                0.50,
                0.75,
            ],
        )

        print(
            f"  mean {pers.mean():.3f}  "
            f"median {np.median(pers):.3f}  "
            f"quartiles "
            f"{np.round(quartiles, 3)}"
        )

    else:

        print(
            "  no interviews with questions"
        )


# ============================================================
# BORDERLINE / DIAGNOSTIC INSPECTION
# ============================================================

def inspect_cases(
    res,
    label=None,
    sim_lo=0.55,
    sim_hi=0.90,
    n=25,
):

    mask = (
        (res["best_sim"] >= sim_lo)
        & (res["best_sim"] < sim_hi)
    )

    if label is not None:

        mask &= (
            res["lay"] == label
        )

    idx = np.where(mask)[0]

    if len(idx) == 0:

        print(
            "\nNo cases found."
        )

        return

    idx = idx[
        np.argsort(
            -res["spine_coverage"][idx]
        )
    ]

    idx = idx[:n]

    print(
        f"\n=== INSPECT "
        f"{res['name'].upper()} "
        f"({label or 'all labels'}) ==="
    )

    for k in idx:

        print(
            "\n" + "-" * 80
        )

        print(
            f"label            : "
            f"{res['lay'][k]}"
        )

        print(
            f"spine similarity : "
            f"{res['best_sim'][k]:.3f}"
        )

        print(
            f"spine coverage   : "
            f"{res['spine_coverage'][k]:.3f}"
        )

        print(
            f"question coverage: "
            f"{res['q_coverage'][k]:.3f}"
        )

        print(
            f"sequence sim     : "
            f"{res['sequence_sim'][k]:.3f}"
        )

        print(
            f"answer overlap   : "
            f"{res['echo'][k]:.3f}"
        )

        print(
            "\nSPINE:"
        )

        print(
            res["nearest_spine"][k]
        )

        print(
            "\nQUESTION:"
        )

        print(
            res["q"][k]
        )

        print(
            "\nANSWER:"
        )

        print(
            res["a"][k][:500]
        )


# ============================================================
# PARSER CHECK
# ============================================================

def check_parser(
    ds,
    n_per_split=3,
):

    for split in (
        "workforce",
        "creatives",
        "scientists",
    ):

        print(
            f"\n\n========== "
            f"PARSER CHECK: {split.upper()} "
            f"=========="
        )

        for i in range(
            min(
                n_per_split,
                len(ds[split]),
            )
        ):

            pairs = qa_pairs(
                ds[split][i]["text"],
                skip_first_assistant_turn=(
                    SKIP_FIRST_ASSISTANT_TURN
                ),
            )

            print(
                f"\nInterview {i}: "
                f"{len(pairs)} pairs"
            )

            for q, a in pairs[:5]:

                print(
                    "\nQ:",
                    q,
                )

                print(
                    "A:",
                    a[:300],
                )


# ============================================================
# RUN ALL SPLITS
# ============================================================

def run_all(ds):

    model = SentenceTransformer(
        MODEL_NAME
    )

    results = {}

    for split in (
        "workforce",
        "creatives",
        "scientists",
    ):

        results[split] = analyze_split(
            split,
            ds[split],
            model,
        )

        report(
            results[split]
        )

    return results


# ============================================================
# LOAD DATA
# ============================================================

ds = load_dataset(
    "Anthropic/AnthropicInterviewer"
)


# ============================================================
# STEP 1: PARSER CHECK
# ============================================================

check_parser(
    ds,
    n_per_split=3,
)


# ============================================================
# STEP 2: FULL ANALYSIS
# ============================================================

results = run_all(ds)


# ============================================================
# STEP 3: DIAGNOSTICS
# ============================================================

inspect_cases(
    results["workforce"],
    sim_lo=0.60,
    sim_hi=0.90,
    n=30,
)

inspect_cases(
    results["workforce"],
    label="personalized",
    sim_lo=0.60,
    sim_hi=0.90,
    n=30,
)

inspect_cases(
    results["workforce"],
    label="branch",
    sim_lo=0.60,
    sim_hi=0.90,
    n=30,
)

inspect_cases(
    results["scientists"],
    label="personalized",
    sim_lo=0.60,
    sim_hi=0.90,
    n=30,
)

inspect_cases(
    results["scientists"],
    label="branch",
    sim_lo=0.60,
    sim_hi=0.90,
    n=30,
)
