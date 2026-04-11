"""
Market Narrative Intelligence — FastAPI Backend
================================================
Wraps the existing scenario engine as a REST API.

Endpoints:
  POST /scenario        — single scenario assessment
  POST /compare         — side-by-side scenario comparison
  POST /crisis          — crisis response comparator
  POST /sector          — three-view sector analysis
  GET  /sectors         — list all sectors and event counts
  GET  /events          — list all events in the database
  GET  /health          — health check

Run locally:
  uvicorn main:app --reload --port 8000

Then open: http://localhost:8000/docs
"""

import os
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Import scenario engine functions ─────────────────────────
from scenario_engine import (
    load_event_database,
    build_event_index,
    find_similar_events,
    find_similar_events_with_sector,
    synthesise_impact,
    split_by_response_strategy,
)


# ── App state — loaded once at startup ───────────────────────
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and event database once at startup."""
    print("Loading scenario engine...")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    events = load_event_database()
    index  = build_event_index(events, model)

    # Build sector index
    sector_index = {}
    for event in events:
        s = event.get("sector", "Unknown")
        sector_index[s] = sector_index.get(s, 0) + 1

    app_state["model"]        = model
    app_state["events"]       = events
    app_state["index"]        = index
    app_state["sector_index"] = sector_index

    print(f"Ready — {len(events)} events loaded across {len(sector_index)} sectors")
    yield
    app_state.clear()


app = FastAPI(
    title="Market Narrative Intelligence API",
    description="Scenario-based market impact analysis using historical FTSE 100 event studies",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow requests from the React frontend (localhost in dev, Vercel in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten this to your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────

class ScenarioRequest(BaseModel):
    scenario: str

class CompareRequest(BaseModel):
    scenario_a: str
    scenario_b: str

class SectorRequest(BaseModel):
    scenario: str
    sector:   str


# ── Helper: format a match list for the API response ─────────

def _f(val):
    """Convert numpy floats to native Python floats for JSON serialisation."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return val

def format_matches(matches):
    result = []
    for score, event in matches:
        m = event.get("metrics", {})
        result.append({
            "headline":        event.get("headline", ""),
            "company":         event.get("company",  ""),
            "ticker":          event.get("ticker",   ""),
            "sector":          event.get("sector",   ""),
            "event_date":      event.get("event_date", ""),
            "similarity":      round(float(score), 4),
            "narrative_tag":   event.get("narrative_tag", ""),
            "response_type":   event.get("response_type",  "proactive"),
            "response_timing": event.get("response_timing", "planned"),
            "crisis_trigger":  event.get("crisis_trigger",  "none"),
            "peak_car_pct":    _f(m.get("peak_car_pct")),
            "final_car_pct":   _f(m.get("final_car_pct")),
            "drift_period_days": _f(m.get("drift_period_days")),
            "volatility_spike":  _f(m.get("volatility_spike")),
            "mean_reversion":    m.get("mean_reversion"),
        })
    return result


def format_synthesis(synthesis, matches):
    if not synthesis:
        return None
    strong = sum(1 for s, _ in matches if s >= 0.50)
    return {
        "direction":           synthesis["direction"],
        "expected_peak_car":   _f(synthesis.get("expected_peak_car")),
        "expected_final_car":  _f(synthesis.get("expected_final_car")),
        "expected_drift_days": _f(synthesis.get("expected_drift_days")),
        "expected_vol_spike":  _f(synthesis.get("expected_vol_spike")),
        "mean_reversion":      synthesis.get("mean_reversion_label"),
        "confidence":          synthesis["confidence"],
        "precedent_count":     len(matches),
        "strong_match_count":  strong,
    }


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "event_count":  len(app_state.get("events", [])),
        "sector_count": len(app_state.get("sector_index", {})),
    }


@app.get("/sectors")
def get_sectors():
    sector_index = app_state.get("sector_index", {})
    return {
        "sectors": [
            {"name": s, "event_count": count}
            for s, count in sorted(sector_index.items(), key=lambda x: -x[1])
        ]
    }


@app.get("/events")
def get_events():
    events = app_state.get("events", [])
    return {
        "count": len(events),
        "events": [
            {
                "headline":      e.get("headline", ""),
                "company":       e.get("company",  ""),
                "ticker":        e.get("ticker",   ""),
                "sector":        e.get("sector",   ""),
                "event_date":    e.get("event_date", ""),
                "narrative_tag": e.get("narrative_tag", ""),
                "response_type": e.get("response_type", "proactive"),
                "final_car_pct": e.get("metrics", {}).get("final_car_pct"),
            }
            for e in events
        ]
    }


@app.post("/scenario")
def run_scenario(request: ScenarioRequest):
    if not request.scenario.strip():
        raise HTTPException(status_code=400, detail="Scenario text is required")

    model  = app_state["model"]
    index  = app_state["index"]

    matches   = find_similar_events(request.scenario, index, model)
    synthesis = synthesise_impact(matches)

    return {
        "scenario":   request.scenario,
        "matches":    format_matches(matches),
        "synthesis":  format_synthesis(synthesis, matches),
    }


@app.post("/compare")
def run_compare(request: CompareRequest):
    if not request.scenario_a.strip() or not request.scenario_b.strip():
        raise HTTPException(status_code=400, detail="Both scenarios are required")

    model = app_state["model"]
    index = app_state["index"]

    matches_a   = find_similar_events(request.scenario_a, index, model)
    matches_b   = find_similar_events(request.scenario_b, index, model)
    synthesis_a = synthesise_impact(matches_a)
    synthesis_b = synthesise_impact(matches_b)

    # Calculate differential
    differential = None
    if synthesis_a and synthesis_b:
        delta_car   = _f(synthesis_a["expected_final_car"] or 0) - _f(synthesis_b["expected_final_car"] or 0)
        delta_drift = _f(synthesis_a["expected_drift_days"] or 0) - _f(synthesis_b["expected_drift_days"] or 0)
        differential = {
            "final_car_delta":   round(float(delta_car), 2),
            "drift_delta":       round(float(delta_drift), 1),
            "outperformer":      "a" if delta_car > 0 else "b" if delta_car < 0 else "equal",
        }

    return {
        "scenario_a":     request.scenario_a,
        "scenario_b":     request.scenario_b,
        "matches_a":      format_matches(matches_a),
        "matches_b":      format_matches(matches_b),
        "synthesis_a":    format_synthesis(synthesis_a, matches_a),
        "synthesis_b":    format_synthesis(synthesis_b, matches_b),
        "differential":   differential,
    }


@app.post("/crisis")
def run_crisis(request: ScenarioRequest):
    if not request.scenario.strip():
        raise HTTPException(status_code=400, detail="Scenario text is required")

    model = app_state["model"]
    index = app_state["index"]

    all_matches           = find_similar_events(request.scenario, index, model)
    immediate, delayed, proactive = split_by_response_strategy(all_matches)

    synth_imm = synthesise_impact(immediate)
    synth_del = synthesise_impact(delayed)
    synth_pro = synthesise_impact(proactive)

    # Speed vs delay differential
    differential = None
    if synth_imm and synth_del:
        delta = _f(synth_imm["expected_final_car"] or 0) - _f(synth_del["expected_final_car"] or 0)
        differential = {
            "immediate_vs_delayed_car": round(float(delta), 2),
            "favours": "immediate" if delta > 0.3 else "delayed" if delta < -0.3 else "neutral",
        }

    # Crisis discount vs proactive
    crisis_discount = None
    if synth_imm and synth_pro:
        discount = _f(synth_imm["expected_final_car"] or 0) - _f(synth_pro["expected_final_car"] or 0)
        crisis_discount = round(float(discount), 2)

    return {
        "scenario":        request.scenario,
        "total_matches":   len(all_matches),
        "immediate":       format_matches(immediate),
        "delayed":         format_matches(delayed),
        "proactive":       format_matches(proactive),
        "synthesis_immediate": format_synthesis(synth_imm, immediate),
        "synthesis_delayed":   format_synthesis(synth_del, delayed),
        "synthesis_proactive": format_synthesis(synth_pro, proactive),
        "differential":        differential,
        "crisis_discount":     crisis_discount,
    }


@app.post("/sector")
def run_sector(request: SectorRequest):
    if not request.scenario.strip():
        raise HTTPException(status_code=400, detail="Scenario text is required")
    if not request.sector.strip():
        raise HTTPException(status_code=400, detail="Sector is required")

    model        = app_state["model"]
    index        = app_state["index"]
    sector_index = app_state["sector_index"]

    # Validate sector
    available = list(sector_index.keys())
    matched_sector = None
    for s in available:
        if request.sector.lower() in s.lower():
            matched_sector = s
            break

    if not matched_sector:
        raise HTTPException(
            status_code=400,
            detail=f"Sector not found. Available: {', '.join(sorted(available))}"
        )

    sector_matches, market_matches = find_similar_events_with_sector(
        request.scenario, index, model, matched_sector
    )
    sector_synth = synthesise_impact(sector_matches)
    market_synth = synthesise_impact(market_matches)

    # Delta calculation
    delta = None
    if sector_synth and market_synth:
        delta_car   = _f(sector_synth["expected_final_car"] or 0) - _f(market_synth["expected_final_car"] or 0)
        delta_drift = _f(sector_synth["expected_drift_days"] or 0) - _f(market_synth["expected_drift_days"] or 0)
        delta = {
            "final_car_delta": round(float(delta_car), 2),
            "drift_delta":     round(float(delta_drift), 1),
            "sector_vs_market": (
                "outperforms" if delta_car > 0.5 else
                "underperforms" if delta_car < -0.5 else
                "similar"
            ),
        }

    return {
        "scenario":       request.scenario,
        "sector":         matched_sector,
        "sector_matches": format_matches(sector_matches),
        "market_matches": format_matches(market_matches),
        "sector_synthesis": format_synthesis(sector_synth, sector_matches),
        "market_synthesis": format_synthesis(market_synth, market_matches),
        "delta":          delta,
    }
