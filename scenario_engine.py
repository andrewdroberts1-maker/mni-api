"""
Market Narrative Intelligence — Scenario Engine
================================================
This is the core of the platform. It answers the question:

  "If we announce X, what is the likely market impact?"

It works in four steps:

  1. You describe a potential decision or announcement in plain English
  2. The engine converts your text into a mathematical vector (an "embedding")
     that captures its meaning — not just the words, but what they signify
  3. It searches the historical event database for the most similar past events,
     using semantic similarity (so "interim CEO appointed" matches
     "leadership transition announced" even with no shared words)
  4. It synthesises the CAR profiles of those matching events into a
     confidence range: expected drift period, likely magnitude, mean reversion

The most powerful feature is SCENARIO COMPARISON — you can describe two
alternative courses of action and see their historical impact profiles
side by side. For example:

  Scenario A: "We announce our CEO is stepping down with no successor named"
  Scenario B: "We announce our CEO is stepping down with a permanent successor"

HOW TO RUN
----------
  python scenario_engine.py

On first run, it will download a small language model (~80MB, one-time only)
used to generate text embeddings. After that it runs entirely offline.

The engine runs in interactive mode — just type your scenario when prompted.
Type 'compare' to run a side-by-side comparison of two scenarios.
Type 'list'    to see all events in the historical database.
Type 'quit'    to exit.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────

# The scenario engine automatically uses the FTSE 20 master database
# if it exists, otherwise falls back to the single-company file.
import os as _os
EVENT_STUDY_FILE          = _os.environ.get(
    "DATABASE_PATH",
    _os.path.join("data", "FTSE20_master_database.json")
)
EVENT_STUDY_FILE_FALLBACK = _os.path.join("data", "VOD_event_study.json")

# GICS sectors available for filtering
SECTORS = [
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
]
# Threshold-based matching — include all events above the quality
# threshold rather than an arbitrary fixed count. This means a rich
# query with 8 strong matches produces a more reliable synthesis than
# a weak query with only 2, and the confidence level reflects this.
MIN_SIMILARITY   = 0.25  # minimum score to be included as a precedent
STRONG_MATCH     = 0.50  # above this = strong match, weighted 3x
MAX_MATCHES      = 8     # soft cap — prevents very broad queries from
                         # including marginally relevant events


# ─────────────────────────────────────────────────────────────
#  STEP 1: Load the historical event database
#  (produced by event_study.py)
# ─────────────────────────────────────────────────────────────

def load_event_database():
    # Try master FTSE 20 database first, fall back to single-company file
    for filepath in [EVENT_STUDY_FILE, EVENT_STUDY_FILE_FALLBACK]:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            events = data.get("events", [])
            source = "FTSE 20 master database" if "FTSE20" in filepath else "single-company database"
            print(f"  ✓  Loaded {len(events)} historical events from {source}")
            return events
    print("\n  ✗  No event database found.")
    print(f"     Looking for: {EVENT_STUDY_FILE}")
    print("     Copy FTSE20_master_database.json into the data/ folder.")
    return []

def load_embedding_model():
    print("\n  Loading semantic embedding model...")
    print("  (First run only: downloading ~80MB model — subsequent runs are instant)")

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("  ✓  Embedding model ready")
        return model
    except ImportError:
        print("  ✗  sentence-transformers not installed.")
        print("     Run:  pip install sentence-transformers")
        return None


def embed_text(model, text):
    """Convert a text string into a numpy embedding vector."""
    return model.encode(text, convert_to_numpy=True)


def cosine_similarity(vec_a, vec_b):
    """
    Measure how similar two vectors are. Returns a value between -1 and 1.
    1.0 = identical meaning, 0.0 = unrelated, -1.0 = opposite meaning.
    """
    dot    = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─────────────────────────────────────────────────────────────
#  STEP 3: Build the searchable index
#
#  For each historical event, we create a rich text description
#  that combines the headline, category, and narrative tag.
#  This gives the embedding more signal to work with.
# ─────────────────────────────────────────────────────────────

def build_event_index(events, model):
    """
    Build semantic search index from events.
    Loads pre-computed embeddings from disk if available —
    much faster on Railway. Falls back to computing if not found.
    """
    EMBEDDINGS_PATH = os.path.join("data", "event_embeddings.npy")
    HEADLINES_PATH  = os.path.join("data", "event_headlines.json")

    headlines = [e.get("headline", "") for e in events]

    # Try loading pre-computed embeddings
    if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(HEADLINES_PATH):
        try:
            saved_embeddings = np.load(EMBEDDINGS_PATH)
            with open(HEADLINES_PATH) as f:
                saved_headlines = json.load(f)

            # Verify alignment — same number of events
            if len(saved_embeddings) == len(events):
                print(f"  ✓  Loaded pre-computed embeddings ({len(events)} events)")
                print(f"  ✓  Index ready — {len(events)} events ready for search")
                return [(event, saved_embeddings[i], headlines[i])
                        for i, event in enumerate(events)]
            else:
                print(f"  ⚠  Embedding count mismatch ({len(saved_embeddings)} vs {len(events)}) — recomputing")
        except Exception as e:
            print(f"  ⚠  Could not load embeddings: {e} — recomputing")

    # Fall back to computing
    print("  Building semantic index of historical events...")
    embeddings = model.encode(headlines, show_progress_bar=False, batch_size=64)
    print(f"  ✓  Index built — {len(events)} events ready for search")
    return [(event, embeddings[i], headlines[i]) for i, event in enumerate(events)]


# ─────────────────────────────────────────────────────────────
#  STEP 4: Search for similar historical events
# ─────────────────────────────────────────────────────────────

def find_similar_events(scenario_text, index, model):
    """
    Given a scenario description, find all events above the quality
    threshold, weighted by similarity. Strong matches (above STRONG_MATCH)
    are counted at 3x weight in the synthesis to reflect their higher
    relevance. Results are capped at MAX_MATCHES to avoid dilution.
    """
    scenario_embedding = embed_text(model, scenario_text)

    scored = []
    for event, event_embedding, event_text in index:
        score = cosine_similarity(scenario_embedding, event_embedding)
        scored.append((score, event))

    # Sort by similarity score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Apply quality threshold — only include genuinely relevant matches
    matches = [(score, event) for score, event in scored if score >= MIN_SIMILARITY]

    # Cap at MAX_MATCHES to avoid including marginally relevant events
    return matches[:MAX_MATCHES]


# ─────────────────────────────────────────────────────────────
#  STEP 4b: Sector-filtered search
#
#  Runs the same semantic search but restricts the index to
#  events from companies in the specified sector only.
#  Returns (sector_matches, market_matches) for three-view output.
# ─────────────────────────────────────────────────────────────

def find_similar_events_with_sector(scenario_text, index, model, sector):
    """
    Run semantic search twice:
      1. Sector-filtered: only events from companies in the target sector
      2. Full market:     all events regardless of sector
    Returns (sector_matches, market_matches)
    """
    scenario_embedding = embed_text(model, scenario_text)

    sector_scored  = []
    market_scored  = []

    for event, event_embedding, event_text in index:
        score = cosine_similarity(scenario_embedding, event_embedding)
        market_scored.append((score, event))
        if event.get("sector", "").lower() == sector.lower():
            sector_scored.append((score, event))

    for scored in (sector_scored, market_scored):
        scored.sort(key=lambda x: x[0], reverse=True)

    sector_matches = [(s, e) for s, e in sector_scored  if s >= MIN_SIMILARITY][:MAX_MATCHES]
    market_matches = [(s, e) for s, e in market_scored  if s >= MIN_SIMILARITY][:MAX_MATCHES]

    return sector_matches, market_matches


# ─────────────────────────────────────────────────────────────
#  STEP 5: Synthesise impact assessment from matched events
#
#  Average the key metrics across matched events, weighted by
#  their similarity score. More similar events carry more weight.
# ─────────────────────────────────────────────────────────────

def synthesise_impact(matches):
    """
    Produce a weighted-average impact estimate from the matched events.
    Returns a dict of synthesised metrics and a confidence note.
    """
    if not matches:
        return None

    total_weight = sum(score for score, _ in matches)
    if total_weight == 0:
        return None

    weighted_peak    = 0.0
    weighted_final   = 0.0
    weighted_drift   = 0.0
    weighted_vol     = 0.0
    vol_count        = 0
    reversion_labels = []

    for score, event in matches:
        # Strong matches carry 3x the weight of threshold-level matches
        adjusted_score = score * 3 if score >= STRONG_MATCH else score
        w       = adjusted_score / sum(
            s * 3 if s >= STRONG_MATCH else s for s, _ in matches
        )
        metrics = event.get("metrics", {})

        weighted_peak  += w * metrics.get("peak_car_pct",  0)
        weighted_final += w * metrics.get("final_car_pct", 0)
        weighted_drift += w * metrics.get("drift_period_days", 0)

        vol = metrics.get("volatility_spike")
        if vol is not None:
            weighted_vol += w * vol
            vol_count    += 1

        reversion_labels.append(metrics.get("mean_reversion", ""))

    # Direction: positive or negative based on weighted final CAR
    direction = "positive" if weighted_final >= 0 else "negative"

    # Confidence: based on both top match score AND number of matches
    top_score    = matches[0][0]
    strong_count = sum(1 for score, _ in matches if score >= STRONG_MATCH)

    if top_score > 0.65 and strong_count >= 3:
        confidence = "high"
    elif top_score > 0.50 or strong_count >= 2:
        confidence = "medium"
    elif top_score > 0.35:
        confidence = "low — treat as indicative only"
    else:
        confidence = "very low — limited historical precedent found"

    # Most common reversion label
    reversion = max(set(reversion_labels), key=reversion_labels.count) if reversion_labels else "unknown"

    return {
        "direction":           direction,
        "expected_peak_car":   round(weighted_peak,  2),
        "expected_final_car":  round(weighted_final, 2),
        "expected_drift_days": round(weighted_drift, 1),
        "expected_vol_spike":  round(weighted_vol,   2) if vol_count > 0 else None,
        "mean_reversion":      reversion,
        "confidence":          confidence,
        "top_match_score":     round(top_score, 3),
        "n_precedents":        len(matches),
    }


# ─────────────────────────────────────────────────────────────
#  STEP 6: Print a formatted single-scenario report
# ─────────────────────────────────────────────────────────────

def print_scenario_report(scenario_text, matches, synthesis):
    width = 62
    print("\n" + "═" * width)
    print("  SCENARIO IMPACT ASSESSMENT")
    print("─" * width)
    print(f"  Input: \"{scenario_text[:70]}{'...' if len(scenario_text) > 70 else ''}\"")
    print("─" * width)

    if not matches:
        print("  No sufficiently similar historical events found.")
        print("  Try rephrasing or expanding your scenario description.")
        print("═" * width)
        return

    strong = sum(1 for score, _ in matches if score >= STRONG_MATCH)
    print(f"\n  {len(matches)} HISTORICAL PRECEDENT(S) FOUND  "
          f"({strong} strong match{'es' if strong != 1 else ''} above {STRONG_MATCH:.0%})\n")
    for i, (score, event) in enumerate(matches, 1):
        m  = event.get("metrics", {})
        rt = event.get("response_type",  "proactive")
        ct = event.get("crisis_trigger", "none")
        response_label = f"reactive — {ct}" if rt == "reactive" else "proactive"
        print(f"  {i}.  {event['headline'][:55]}")
        print(f"       Date      : {event.get('event_date', 'unknown')}")
        print(f"       Similarity: {score:.1%}")
        print(f"       Response  : {response_label}")
        print(f"       Peak CAR  : {m.get('peak_car_pct', 'n/a'):+}%")
        print(f"       Final CAR : {m.get('final_car_pct', 'n/a'):+}%")
        print(f"       Drift     : {m.get('drift_period_days', 'n/a')} trading days")
        print(f"       Reversion : {m.get('mean_reversion', 'n/a')}")
        print()

    if synthesis:
        print("─" * width)
        print("  SYNTHESISED OUTLOOK\n")
        direction_arrow = "▲" if synthesis["direction"] == "positive" else "▼"
        print(f"  Direction            {direction_arrow}  {synthesis['direction'].upper()}")
        print(f"  Expected peak CAR    {synthesis['expected_peak_car']:+.1f}%")
        print(f"  Expected final CAR   {synthesis['expected_final_car']:+.1f}%")
        print(f"  Expected drift       {synthesis['expected_drift_days']:.0f} trading days")
        if synthesis["expected_vol_spike"]:
            print(f"  Volatility spike     {synthesis['expected_vol_spike']:.1f}× normal")
        print(f"  Mean reversion       {synthesis['mean_reversion']}")
        print(f"  Confidence           {synthesis['confidence']}")
        print(f"  Based on             {synthesis['n_precedents']} precedent(s)")
    print("═" * width)


# ─────────────────────────────────────────────────────────────
#  STEP 7: Side-by-side scenario comparison
#  This is the key feature for decisions like:
#  "Interim CEO now vs named successor at announcement"
# ─────────────────────────────────────────────────────────────

def print_comparison_report(scenario_a, scenario_b,
                             matches_a, matches_b,
                             synth_a,   synth_b):
    width = 62
    print("\n" + "═" * width)
    print("  SCENARIO COMPARISON")
    print("─" * width)
    print(f"  A: \"{scenario_a[:55]}{'...' if len(scenario_a) > 55 else ''}\"")
    print(f"  B: \"{scenario_b[:55]}{'...' if len(scenario_b) > 55 else ''}\"")
    print("─" * width)

    def fmt(val, suffix="", plus=False):
        if val is None:
            return "n/a"
        if plus:
            return f"{val:+.1f}{suffix}"
        return f"{val:.1f}{suffix}"

    # Build comparison table
    rows = [
        ("Direction",        synth_a["direction"].upper() if synth_a else "n/a",
                             synth_b["direction"].upper() if synth_b else "n/a"),
        ("Peak CAR",         fmt(synth_a["expected_peak_car"]  if synth_a else None, "%", True),
                             fmt(synth_b["expected_peak_car"]  if synth_b else None, "%", True)),
        ("Final CAR",        fmt(synth_a["expected_final_car"] if synth_a else None, "%", True),
                             fmt(synth_b["expected_final_car"] if synth_b else None, "%", True)),
        ("Drift period",     fmt(synth_a["expected_drift_days"] if synth_a else None, " days"),
                             fmt(synth_b["expected_drift_days"] if synth_b else None, " days")),
        ("Confidence",       synth_a["confidence"] if synth_a else "n/a",
                             synth_b["confidence"] if synth_b else "n/a"),
        ("Top precedent",    matches_a[0][1]["headline"][:30] + "..." if matches_a else "none",
                             matches_b[0][1]["headline"][:30] + "..." if matches_b else "none"),
    ]

    print(f"\n  {'METRIC':<22} {'SCENARIO A':<18} {'SCENARIO B'}")
    print(f"  {'──────':<22} {'──────────':<18} {'──────────'}")
    for label, val_a, val_b in rows:
        print(f"  {label:<22} {val_a:<18} {val_b}")

    # Differential: which scenario has a better expected outcome?
    print("\n" + "─" * width)
    print("  DIFFERENTIAL ANALYSIS\n")

    if synth_a and synth_b:
        final_diff = synth_b["expected_final_car"] - synth_a["expected_final_car"]
        drift_diff = synth_b["expected_drift_days"] - synth_a["expected_drift_days"]

        if abs(final_diff) < 0.3:
            print("  Final CAR   Similar expected outcomes — historical")
            print("              precedents do not strongly favour either path.")
        elif final_diff > 0:
            print(f"  Final CAR   Scenario B historically outperforms A")
            print(f"              by approximately {final_diff:+.1f}% in final CAR.")
        else:
            print(f"  Final CAR   Scenario A historically outperforms B")
            print(f"              by approximately {abs(final_diff):.1f}% in final CAR.")

        if abs(drift_diff) < 1:
            print("  Drift       Both scenarios show similar market absorption periods.")
        elif drift_diff > 0:
            print(f"  Drift       Scenario B drives a longer period of price movement")
            print(f"              ({drift_diff:+.1f} days more than Scenario A).")
        else:
            print(f"  Drift       Scenario A drives a longer period of price movement")
            print(f"              ({abs(drift_diff):.1f} days more than Scenario B).")

    strong_a = sum(1 for s, _ in matches_a if s >= STRONG_MATCH) if matches_a else 0
    strong_b = sum(1 for s, _ in matches_b if s >= STRONG_MATCH) if matches_b else 0
    print(f"\n  Precedents: A={len(matches_a) if matches_a else 0} "
          f"({strong_a} strong)  B={len(matches_b) if matches_b else 0} "
          f"({strong_b} strong)")
    print("  Confidence reflects both match quality and precedent count.")
    print("═" * width)


def draw_comparison_chart(scenario_a, scenario_b, matches_a, matches_b, output_dir):
    """
    Draw a side-by-side CAR chart for two scenarios, overlaying the
    matched historical events for each.
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.patch.set_facecolor("#f8f9fa")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax, scenario_text, matches, label in [
        (ax_a, scenario_a, matches_a, "A"),
        (ax_b, scenario_b, matches_b, "B"),
    ]:
        ax.set_facecolor("#ffffff")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#dddddd")
        ax.spines["bottom"].set_color("#dddddd")
        ax.tick_params(colors="#555555", labelsize=9)
        ax.axvline(x=0, color="#888888", linewidth=1, linestyle="--", zorder=2)
        ax.axhline(y=0, color="#cccccc", linewidth=0.8)
        ax.set_title(
            f"Scenario {label}: {scenario_text[:45]}{'...' if len(scenario_text) > 45 else ''}",
            fontsize=9, color="#333333", pad=10
        )
        ax.set_xlabel("Trading days relative to announcement", fontsize=9, color="#555555")
        if label == "A":
            ax.set_ylabel("Cumulative Abnormal Return (%)", fontsize=9, color="#555555")

        if not matches:
            ax.text(0.5, 0.5, "No precedents found", transform=ax.transAxes,
                    ha="center", va="center", color="#999999", fontsize=10)
            continue

        for i, (score, event) in enumerate(matches):
            daily = event.get("daily_results", [])
            if not daily:
                continue
            days = [d["relative_day"] for d in daily]
            cars = [d["cumulative_ar"]  for d in daily]
            color = colors[i % len(colors)]
            alpha = max(0.15, 0.9 - i * 0.15)  # clamp — never below 0.15
            ax.plot(days, cars, color=color, linewidth=1.8, alpha=alpha,
                    label=f"{event['headline'][:30]}... ({score:.0%})")

        ax.legend(fontsize=7, loc="lower right", framealpha=0.7)

    plt.suptitle("Scenario Comparison — Historical CAR Profiles",
                 fontsize=11, color="#333333", y=1.01)
    plt.tight_layout(pad=2.0)

    filepath = os.path.join(output_dir, "scenario_comparison.png")
    plt.savefig(filepath, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  ✓  Comparison chart saved → data/charts/scenario_comparison.png")
    return filepath


# ─────────────────────────────────────────────────────────────
#  THREE-VIEW SECTOR REPORT
#
#  Shows: (1) sector-only precedents, (2) full market precedents,
#  (3) the delta between them — the most actionable output for a
#  client asking "does my industry behave differently?"
# ─────────────────────────────────────────────────────────────

def print_sector_report(scenario_text, sector,
                        sector_matches, market_matches,
                        sector_synth, market_synth):
    width = 66
    print("\n" + "=" * width)
    print("  SECTOR INTELLIGENCE — THREE-VIEW ANALYSIS")
    print("-" * width)
    truncated = (scenario_text[:60] + "...") if len(scenario_text) > 60 else scenario_text
    print(f"  Scenario : {truncated!r}")
    print(f"  Sector   : {sector}")
    print("-" * width)

    def fmt(val, suffix="", plus=False):
        if val is None:
            return "n/a"
        return f"{val:+.1f}{suffix}" if plus else f"{val:.1f}{suffix}"

    # ── View 1: sector only ──────────────────────────────────
    print(f"\n  VIEW 1 — {sector.upper()} SECTOR ONLY")
    sector_event_count = sum(1 for _, e in market_matches if e.get("sector") == sector)
    if not sector_matches:
        print(f"  No precedents found in {sector} with current data.")
        print(f"  ({sector_event_count} sector events in database — more data needed)")
    else:
        strong = sum(1 for s, _ in sector_matches if s >= STRONG_MATCH)
        print(f"  {len(sector_matches)} precedent(s)  |  {strong} strong match(es) above {STRONG_MATCH:.0%}")
        for i, (score, event) in enumerate(sector_matches[:3], 1):
            m  = event.get("metrics", {})
            rt = event.get("response_type",  "proactive")
            ct = event.get("crisis_trigger", "none")
            response_label = f"reactive — {ct}" if rt == "reactive" else "proactive"
            print(f"  {i}.  {event['headline'][:52]}")
            print(f"       {event.get('company','?'):<25} "
                  f"Similarity: {score:.1%}  "
                  f"Final CAR: {m.get('final_car_pct', 0):+.1f}%  "
                  f"Response: {response_label}")

    if sector_synth:
        print("\n  Sector synthesised outlook:")
        arrow = "▲" if sector_synth["direction"] == "positive" else "▼"
        print(f"    Direction      {arrow}  {sector_synth['direction'].upper()}")
        print(f"    Final CAR      {fmt(sector_synth['expected_final_car'], '%', True)}")
        print(f"    Drift          {fmt(sector_synth['expected_drift_days'], ' days')}")
        print(f"    Confidence     {sector_synth['confidence']}")

    # ── View 2: full market ──────────────────────────────────
    print("\n  VIEW 2 — FULL MARKET (ALL SECTORS)")
    if market_synth:
        strong = sum(1 for s, _ in market_matches if s >= STRONG_MATCH)
        print(f"  {len(market_matches)} precedent(s)  |  {strong} strong match(es)")
        arrow = "▲" if market_synth["direction"] == "positive" else "▼"
        print(f"  Direction      {arrow}  {market_synth['direction'].upper()}")
        print(f"  Final CAR      {fmt(market_synth['expected_final_car'], '%', True)}")
        print(f"  Drift          {fmt(market_synth['expected_drift_days'], ' days')}")
        print(f"  Confidence     {market_synth['confidence']}")
    else:
        print(f"  No precedents found above the similarity threshold ({MIN_SIMILARITY:.0%}).")
        print(f"  Try rephrasing with more specific language, or this scenario type")
        print(f"  may need additional seed events in the database.")

    # ── View 3: delta ────────────────────────────────────────
    print("\n  VIEW 3 — SECTOR DELTA (sector vs full market)")
    if sector_synth and market_synth:
        delta_car   = (sector_synth["expected_final_car"]  or 0) - (market_synth["expected_final_car"]  or 0)
        delta_drift = (sector_synth["expected_drift_days"] or 0) - (market_synth["expected_drift_days"] or 0)

        if abs(delta_car) < 0.5:
            print(f"  Final CAR   {sector} behaves similarly to the market overall.")
        elif delta_car > 0:
            print(f"  Final CAR   {sector} outperforms market by {delta_car:+.1f}% for this scenario.")
        else:
            print(f"  Final CAR   {sector} underperforms market by {abs(delta_car):.1f}% for this scenario.")

        if abs(delta_drift) < 0.5:
            print(f"  Drift       Market absorption speed is similar across sectors.")
        elif delta_drift > 0:
            print(f"  Drift       {sector} takes {delta_drift:+.1f} days longer to absorb this news.")
        else:
            print(f"  Drift       {sector} absorbs this news {abs(delta_drift):.1f} days faster.")

        if not sector_matches:
            print(f"\n  Note: Sector view is empty — delta is not meaningful.")
            print(f"  The Ticker Advanced plan upgrade will unlock this view.")
        elif abs(delta_car) < 0.5 and abs(delta_drift) < 0.5:
            print(f"\n  No material sector difference detected at current data volume.")
            print(f"  This view will strengthen with the Advanced plan data upgrade.")
        else:
            print(f"\n  {sector} shows a distinctive pattern for this scenario type.")
    else:
        print(f"  Insufficient data for delta calculation.")

    print("=" * width)

def list_events(events):
    print(f"\n  {'#':<4} {'DATE':<12} {'HEADLINE':<50} {'PEAK CAR':>8}")
    print(f"  {'─'*4} {'─'*12} {'─'*50} {'─'*8}")
    for i, e in enumerate(events, 1):
        m    = e.get("metrics", {})
        peak = m.get("peak_car_pct", 0)
        print(f"  {i:<4} {e.get('event_date','?'):<12} {e['headline'][:50]:<50} {peak:>+7.1f}%")
    print()


def split_by_response_strategy(matches):
    """Split matched events into immediate, delayed and proactive groups."""
    immediate = []
    delayed   = []
    proactive = []
    for score, event in matches:
        rt     = event.get("response_type",  "proactive")
        timing = event.get("response_timing", "planned")
        if rt == "reactive" and timing == "immediate":
            immediate.append((score, event))
        elif rt == "reactive" and timing == "delayed":
            delayed.append((score, event))
        else:
            proactive.append((score, event))
    return immediate, delayed, proactive


def print_crisis_report(scenario_text, all_matches,
                        immediate, delayed, proactive,
                        synth_imm, synth_del, synth_pro):
    width = 70
    print("\n" + "=" * width)
    print("  CRISIS RESPONSE COMPARATOR")
    print("-" * width)
    truncated = (scenario_text[:62] + "...") if len(scenario_text) > 62 else scenario_text
    print(f"  Scenario : {truncated!r}")
    print(f"  Total precedents found: {len(all_matches)}")
    print("-" * width)

    def fmt(val, suffix="", plus=False):
        if val is None: return "n/a"
        return f"{val:+.1f}{suffix}" if plus else f"{val:.1f}{suffix}"

    def col_lines(synth, matches, label):
        if not synth or not matches:
            return [label, "No precedents", "in database", "", "", ""]
        arrow = "▲" if synth["direction"] == "positive" else "▼"
        conf  = synth["confidence"]
        conf_short = ("very low" if "very low" in conf else
                      "low"      if "low"      in conf else
                      "medium"   if "medium"   in conf else "high")
        return [
            label,
            f"{len(matches)} precedent(s)",
            f"Direction  {arrow} {synth['direction'].upper()}",
            f"Final CAR  {fmt(synth['expected_final_car'], '%', True)}",
            f"Drift      {fmt(synth['expected_drift_days'], ' days')}",
            f"Confidence {conf_short}",
        ]

    c1 = col_lines(synth_imm, immediate, "IMMEDIATE STATEMENT")
    c2 = col_lines(synth_del, delayed,   "DELAYED RESPONSE")
    c3 = col_lines(synth_pro, proactive, "PROACTIVE / CONTROL")

    col_w = 23
    print()
    for r1, r2, r3 in zip(c1, c2, c3):
        print(f"  {r1:<{col_w}}  {r2:<{col_w}}  {r3:<{col_w}}")

    print("\n" + "-" * width)
    print("  DIFFERENTIAL ANALYSIS\n")

    if synth_imm and synth_del:
        delta = (synth_imm["expected_final_car"] or 0) - (synth_del["expected_final_car"] or 0)
        if abs(delta) < 0.3:
            print("  Speed vs delay  No material CAR difference between strategies.")
        elif delta > 0:
            print(f"  Speed vs delay  Immediate outperforms delayed by {delta:+.1f}% final CAR.")
            print(f"                  Historical evidence favours acting quickly.")
        else:
            print(f"  Speed vs delay  Delayed outperforms immediate by {abs(delta):.1f}% final CAR.")
            print(f"                  Historical evidence suggests waiting may preserve more value.")
    else:
        print("  Speed vs delay  Insufficient data — need more reactive precedents")
        print("                  of each timing type for a reliable comparison.")

    if synth_imm and synth_pro:
        delta_v_pro = (synth_imm["expected_final_car"] or 0) - (synth_pro["expected_final_car"] or 0)
        if abs(delta_v_pro) < 0.3:
            print("  vs control      Reactive events resolve similarly to proactive announcements.")
        elif delta_v_pro > 0:
            print(f"  vs control      Reactive events outperform proactive by {delta_v_pro:+.1f}%.")
        else:
            print(f"  vs control      Reactive events underperform proactive by {abs(delta_v_pro):.1f}%,")
            print(f"                  consistent with a crisis discount vs planned announcements.")

    reactive_matches = sorted(immediate + delayed, key=lambda x: -x[0])
    if reactive_matches:
        print("\n  REACTIVE PRECEDENTS USED:")
        for score, event in reactive_matches[:6]:
            m       = event.get("metrics", {})
            timing  = event.get("response_timing", "planned")
            trigger = event.get("crisis_trigger", "none")
            rt      = event.get("response_type", "proactive")
            label   = f"{timing} — {trigger}" if rt == "reactive" else "proactive"
            print(f"    {event['headline'][:52]}")
            print(f"      {event.get('company','?'):<22} "
                  f"Sim: {score:.1%}  CAR: {m.get('final_car_pct',0):+.1f}%  [{label}]")

    if len(all_matches) < 4:
        print("\n  Note: Low precedent count — results are indicative only.")
        print("  The crisis module will strengthen as more reactive events")
        print("  are added and the Advanced plan data upgrade is applied.")

    print("=" * width)


def run_interactive(events, index, model):
    # Build sector index before the loop so all commands can use it
    sector_index = {}
    for event in events:
        s = event.get("sector", "Unknown")
        sector_index[s] = sector_index.get(s, 0) + 1

    print("\n" + "=" * 66)
    print("  SCENARIO ENGINE — Interactive Mode")
    print("  Commands: 'compare', 'crisis', 'sector', 'sectors', 'list', 'quit'")
    print("  Or just type a scenario description to assess it.")
    print("=" * 66)

    os.makedirs(os.path.join("data", "charts"), exist_ok=True)

    while True:
        print()
        user_input = input("  Enter scenario (or command): ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("\n  Exiting scenario engine.\n")
            break

        elif user_input.lower() == "list":
            list_events(events)

        elif user_input.lower() == "crisis":
            print("\n  CRISIS RESPONSE COMPARATOR")
            print("  Describe your crisis situation. The engine will compare three")
            print("  response strategies: immediate statement, delayed response,")
            print("  and proactive/control (planned announcements for context).\n")
            scenario_text = input("  Describe the crisis situation:\n  > ").strip()
            if not scenario_text:
                continue

            print("\n  Searching for precedents and splitting by response strategy...")
            all_matches = find_similar_events(scenario_text, index, model)
            immediate, delayed, proactive = split_by_response_strategy(all_matches)

            synth_imm = synthesise_impact(immediate)
            synth_del = synthesise_impact(delayed)
            synth_pro = synthesise_impact(proactive)

            print_crisis_report(
                scenario_text, all_matches,
                immediate, delayed, proactive,
                synth_imm, synth_del, synth_pro
            )

        elif user_input.lower() == "sectors":
            print("\n  Available sectors and event counts:")
            for s, count in sorted(sector_index.items(), key=lambda x: -x[1]):
                print(f"    {s:<32} {count:>3} events")
            print()

        elif user_input.lower() == "sector":
            print("\n  SECTOR INTELLIGENCE MODE")
            print("  Compare how your sector responds vs the broader market.\n")
            print("  Available sectors:")
            for i, s in enumerate(sorted(sector_index.keys()), 1):
                count = sector_index.get(s, 0)
                print(f"    {i:>2}.  {s:<32} ({count} events)")
            print()
            sector_input = input("  Enter sector name (or number): ").strip()

            # Accept either number or name
            sorted_sectors = sorted(sector_index.keys())
            chosen_sector = None
            if sector_input.isdigit():
                idx = int(sector_input) - 1
                if 0 <= idx < len(sorted_sectors):
                    chosen_sector = sorted_sectors[idx]
            else:
                for s in sorted_sectors:
                    if sector_input.lower() in s.lower():
                        chosen_sector = s
                        break

            if not chosen_sector:
                print(f"  Sector not recognised. Type \'sectors\' to see the list.")
                continue

            scenario_text = input(f"\n  Describe your scenario for {chosen_sector}:\n  > ").strip()
            if not scenario_text:
                continue

            print("\n  Running three-view analysis...")
            sector_matches, market_matches = find_similar_events_with_sector(
                scenario_text, index, model, chosen_sector
            )
            sector_synth = synthesise_impact(sector_matches)
            market_synth = synthesise_impact(market_matches)

            print_sector_report(
                scenario_text, chosen_sector,
                sector_matches, market_matches,
                sector_synth, market_synth
            )

        elif user_input.lower() == "compare":
            print("\n  SCENARIO COMPARISON MODE")
            print("  Describe two alternative courses of action.\n")
            scenario_a = input("  Scenario A — describe the first option:\n  > ").strip()
            scenario_b = input("\n  Scenario B — describe the second option:\n  > ").strip()

            if not scenario_a or not scenario_b:
                print("  Both scenarios must be filled in.")
                continue

            print("\n  Searching for precedents...")
            matches_a  = find_similar_events(scenario_a, index, model)
            matches_b  = find_similar_events(scenario_b, index, model)
            synthesis_a = synthesise_impact(matches_a)
            synthesis_b = synthesise_impact(matches_b)

            print_comparison_report(
                scenario_a, scenario_b,
                matches_a,  matches_b,
                synthesis_a, synthesis_b,
            )

            draw_comparison_chart(
                scenario_a, scenario_b,
                matches_a,  matches_b,
                os.path.join("data", "charts"),
            )

        else:
            # Single scenario assessment
            print("\n  Searching for precedents...")
            matches   = find_similar_events(user_input, index, model)
            synthesis = synthesise_impact(matches)
            print_scenario_report(user_input, matches, synthesis)


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  Market Narrative Intelligence — Scenario Engine")
    print("=" * 62)

    # Load historical event database
    events = load_event_database()
    if not events:
        return

    # Load embedding model
    model = load_embedding_model()
    if model is None:
        return

    # Build searchable index
    index = build_event_index(events, model)

    # Run interactive shell
    run_interactive(events, index, model)


if __name__ == "__main__":
    main()
