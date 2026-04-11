"""
Market Narrative Intelligence — Narrative Classifier
=====================================================
Automatically tags RNS announcements with a narrative category
using zero-shot semantic classification.

Each narrative category is represented as a descriptive phrase.
We embed both the event text and the category descriptions, then
assign the category whose embedding is most similar to the event.

This approach runs entirely offline using the same sentence-transformers
model already installed for the scenario engine — no additional
downloads or API keys required.

NARRATIVE TAXONOMY
------------------
The ten categories mirror the tags used in the seed events file,
ensuring that auto-tagged live events are directly comparable to
manually curated historical events in the scenario engine.

  financial_results      — earnings releases, revenue updates, guidance
  profit_warning         — guidance cuts, earnings misses, demand warnings
  leadership_change      — CEO/CFO appointments, planned successions
  leadership_instability — abrupt departures, resignations, interim roles
  ma_activity            — acquisitions, mergers, bids, disposals, JVs
  strategic_shift        — strategy resets, portfolio pivots, new direction
  restructuring          — job cuts, demergers, cost programmes, simplification
  regulatory_pressure    — FCA/CMA investigations, enforcement, litigation
  dividend_cut           — dividend reductions, suspensions, rebases
  reputational_crisis    — misconduct, governance failures, scandal

USAGE
-----
  from classifier import NarrativeClassifier

  clf = NarrativeClassifier()          # loads model (cached after first run)
  tag = clf.classify("Headline text", body="Optional body text")
  print(tag)  # e.g. "profit_warning"

  # Batch classify a list of events
  events = clf.classify_events(events_list)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────
#  NARRATIVE TAXONOMY
#
#  Each category is defined by multiple descriptive phrases.
#  Using several phrases per category improves accuracy —
#  the event embedding is compared against the average of all
#  phrase embeddings for each category.
# ─────────────────────────────────────────────────────────────

NARRATIVE_CATEGORIES = {
    "financial_results": [
        "full year results earnings revenue profit announced",
        "half year interim results financial performance reported",
        "quarterly trading update revenue growth in line with guidance",
        "annual results profit before tax dividend declared",
        "preliminary results full year financial summary",
    ],
    "profit_warning": [
        "profit warning guidance cut earnings below expectations",
        "revenue below expectations weaker than expected demand warning",
        "company lowers full year guidance profit outlook reduced",
        "trading update disappointing performance market consensus missed",
        "adjusted profit guidance lowered challenging trading conditions",
    ],
    "leadership_change": [
        "CEO appointed new chief executive named permanent successor",
        "chief executive officer appointed following search process",
        "new group chief executive joins leadership succession",
        "CFO chief financial officer appointed promoted board change",
        "permanent CEO named following orderly succession planning",
    ],
    "leadership_instability": [
        "CEO resigns steps down immediately no successor named interim",
        "chief executive abrupt departure resignation with immediate effect",
        "CEO ousted board pressure interim chief executive appointed search",
        "sudden leadership departure interim replacement search underway",
        "executive resignation misconduct governance failure immediate effect",
    ],
    "ma_activity": [
        "acquisition agreed merger announcement transaction signed",
        "company agrees to acquire takeover bid offer announced",
        "merger combination strategic transaction regulatory approval",
        "disposal sale of business divests subsidiary completes transaction",
        "takeover approach received bid rejected strategic review",
    ],
    "strategic_shift": [
        "strategy reset new strategic direction announced review",
        "strategic pivot capital allocation framework revised targets",
        "new medium term strategy investor day targets announced",
        "business model change strategic priorities reshaping portfolio",
        "strategic review announced new direction transformation plan",
    ],
    "restructuring": [
        "restructuring programme job cuts workforce reduction announced",
        "demerger separation spin off standalone business created",
        "cost reduction programme savings target headcount reduction",
        "operational review simplification transformation programme",
        "business separation disposal non-core assets streamlining",
    ],
    "regulatory_pressure": [
        "FCA investigation regulatory enforcement action announced fine",
        "CMA competition authority probe antitrust regulatory scrutiny",
        "litigation legal action regulatory penalty provision announced",
        "compliance failure mis-selling investigation regulatory risk charge",
        "regulator inquiry enforcement notice securities over-issuance breach",
        "regulatory sanction penalty over-issuance compliance breach provision",
    ],
    "reputational_crisis": [
        "CEO resigns misconduct personal relationships colleagues scandal",
        "reputational crisis misconduct allegation governance failure board",
        "data breach customer harm public backlash media pressure resignation",
        "ethical failure debanking account closure political controversy",
        "forced resignation media scandal improper disclosure confidential",
    ],
    "dividend_cut": [
        "dividend cut reduced rebased suspended lower payout",
        "dividend per share reduced capital preservation shareholder returns",
        "dividend suspended scrapped eliminated to fund investment",
        "total dividend cut to preserve balance sheet capital",
        "dividend policy rebased lower to support investment programme",
    ],

}


class NarrativeClassifier:
    """
    Zero-shot narrative classifier using semantic similarity.
    Classifies event text into one of ten narrative categories.
    """

    def __init__(self, model=None):
        """
        Initialise the classifier. Optionally pass an existing
        SentenceTransformer model to avoid reloading it.
        """
        self._model   = model
        self._anchors = None   # pre-computed category anchor embeddings

    def _load_model(self):
        """Load the embedding model (cached after first call)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print("  Loading classifier model...")
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            print("  ✓  Classifier ready")
        return self._model

    def _build_anchors(self):
        """
        Pre-compute the average embedding for each narrative category.
        Called once and cached — subsequent calls are instant.
        """
        if self._anchors is not None:
            return self._anchors

        model = self._load_model()
        anchors = {}

        for category, phrases in NARRATIVE_CATEGORIES.items():
            embeddings = model.encode(phrases, convert_to_numpy=True)
            # Average across all phrases for a robust category representation
            anchors[category] = embeddings.mean(axis=0)

        self._anchors = anchors
        return anchors

    def classify(self, headline, body="", threshold=0.20):
        """
        Classify a single event by headline and optional body text.

        Parameters
        ----------
        headline  : str  — the event headline
        body      : str  — optional body text (improves accuracy)
        threshold : float — minimum similarity to assign a tag
                           (below this, returns "unclassified")

        Returns
        -------
        str : narrative tag, e.g. "profit_warning"
        """
        model   = self._load_model()
        anchors = self._build_anchors()

        # Combine headline and first 200 chars of body for classification
        body_snippet = body[:200] if body else ""
        text = f"{headline}. {body_snippet}".strip(". ")

        event_embedding = model.encode(text, convert_to_numpy=True)

        # Cosine similarity against each category anchor
        best_tag   = "unclassified"
        best_score = threshold

        for category, anchor in anchors.items():
            dot    = np.dot(event_embedding, anchor)
            norm_e = np.linalg.norm(event_embedding)
            norm_a = np.linalg.norm(anchor)
            if norm_e == 0 or norm_a == 0:
                continue
            score = dot / (norm_e * norm_a)
            if score > best_score:
                best_score = score
                best_tag   = category

        return best_tag

    def classify_with_scores(self, headline, body=""):
        """
        Like classify() but returns all category scores sorted by confidence.
        Useful for debugging and understanding borderline cases.
        """
        model   = self._load_model()
        anchors = self._build_anchors()

        body_snippet = body[:200] if body else ""
        text = f"{headline}. {body_snippet}".strip(". ")
        event_embedding = model.encode(text, convert_to_numpy=True)

        scores = {}
        for category, anchor in anchors.items():
            dot    = np.dot(event_embedding, anchor)
            norm_e = np.linalg.norm(event_embedding)
            norm_a = np.linalg.norm(anchor)
            if norm_e > 0 and norm_a > 0:
                scores[category] = dot / (norm_e * norm_a)

        return sorted(scores.items(), key=lambda x: -x[1])

    def classify_events(self, events):
        """
        Classify a list of events in batch, updating narrative_tag
        only where it is currently None or missing.
        Preserves any manually set tags from the seed events file.

        Returns the updated list.
        """
        # Pre-build anchors once for efficiency
        self._build_anchors()

        tagged   = 0
        skipped  = 0

        for event in events:
            # Skip events that already have a manually set tag
            existing_tag = event.get("narrative_tag")
            if existing_tag and existing_tag != "unclassified":
                skipped += 1
                continue

            headline = event.get("headline", "")
            body     = event.get("body", "")

            if not headline:
                continue

            tag = self.classify(headline, body)
            event["narrative_tag"] = tag
            tagged += 1

        return events, tagged, skipped


# ─────────────────────────────────────────────────────────────
#  STANDALONE VALIDATION
#
#  Run this file directly to validate the classifier against
#  the seed events — a quick sanity check of accuracy.
# ─────────────────────────────────────────────────────────────

def validate_against_seeds():
    import json

    print("\n" + "=" * 60)
    print("  Narrative Classifier — Validation Against Seed Events")
    print("=" * 60)

    with open("seed_events_ftse20.json") as f:
        seeds = json.load(f)

    clf = NarrativeClassifier()

    correct = 0
    wrong   = 0
    errors  = []

    for key, events in seeds.items():
        if key in ("description", "generated"):
            continue
        if not isinstance(events, list):
            continue

        for event in events:
            true_tag = event.get("narrative_tag", "")
            if not true_tag:
                continue

            predicted = clf.classify(
                event.get("headline", ""),
                event.get("body", "")
            )

            if predicted == true_tag:
                correct += 1
            else:
                wrong += 1
                errors.append({
                    "headline":  event.get("headline", "")[:55],
                    "true":      true_tag,
                    "predicted": predicted,
                })

    total    = correct + wrong
    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\n  Results: {correct}/{total} correct ({accuracy:.0f}% accuracy)\n")

    if errors:
        print(f"  Misclassified events ({len(errors)}):")
        for e in errors:
            print(f"    {e['headline'][:50]}")
            print(f"      True: {e['true']:<28} Predicted: {e['predicted']}")

    print("\n" + "=" * 60)
    return accuracy


if __name__ == "__main__":
    validate_against_seeds()
