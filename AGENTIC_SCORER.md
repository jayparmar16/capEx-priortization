# Deep Dive: The Agentic Scorer (`agentic_scorer.py`)

This document explains the mechanics of the baseline multi-agent system used to prioritize properties for hardening capital.

## Design Choice: Deterministic Simulation vs. Live LLM

In a production environment, enterprise AI governance systems often use large language models (LLMs like GPT-4 or Claude 3) to parse unstructured risk reports (e.g., a PDF Phase 1 Environmental Site Assessment or a complex insurance policy).

**For the purposes of this MVP and to ensure the pipeline runs locally without requiring API keys or incurring costs, `agentic_scorer.py` simulates LLM agents using deterministic Python functions.**

We still use the `LangGraph` framework to orchestrate the state graph and route the data, mimicking the architecture of a true multi-agent system.

If this were plugged into a live LLM (via `langchain`), the "base prompts" for these agents would look like the examples provided below.

---

## 1. The Risk Assessor Agent

**What it does:** Evaluates the physical vulnerability of the property.
**Data Used:** `wind_risk_mph` (ASCE 7 proxy) and `flood_zone` (FEMA proxy).
**The Math:**
* `wind_score` = `(wind_risk_mph - 100) / 100` (normalizing against a baseline)
* `flood_score` = `1.0` for VE (High Hazard), `0.6` for AE, `0.1` for X (Minimal).
* `physical_risk_score` = `(wind_score * 0.5) + (flood_score * 0.5)`

**Simulated Base Prompt:**
> "You are an expert structural engineer and risk assessor. Review the provided FEMA flood zone and ASCE 7 wind exposure data for the following commercial property. Output a normalized `physical_risk_score` between 0.0 and 1.5, weighting wind and flood equally. High coastal hazard zones (VE) and winds over 150mph should approach the maximum score."

---

## 2. The Financial Analyst Agent

**What it does:** Evaluates the financial exposure and value at risk if the property is damaged.
**Data Used:** `noi` (Net Operating Income) and `insurance_premium`.
**The Math:**
* `noi_score` = `min(noi / 2,000,000, 1.0)` (Caps out at $2M NOI)
* `ins_score` = `min(insurance_premium / 500,000, 1.0)`
* `financial_exposure_score` = `(noi_score * 0.6) + (ins_score * 0.4)`

**Simulated Base Prompt:**
> "You are a commercial real estate financial analyst. Review the Net Operating Income and annual insurance premium for the following property. Output a normalized `financial_exposure_score` between 0.0 and 1.0. Weight the NOI heavily (60%) as it represents the cash flow at risk, but also consider the insurance burden (40%) as an indicator of carrier pricing pressure."

---

## 3. The Prioritization Scorer Agent

**What it does:** The final decision node. It takes the output from the Risk Assessor and the Financial Analyst to create the final "fund-first" list.
**Data Used:** `physical_risk_score` and `financial_exposure_score`.
**The Math:**
* `final_priority_score` = `(financial_exposure_score * 0.6) + (physical_risk_score * 0.4)`
* The dataset is sorted descending by this final score.
* The top 30% of properties are flagged as `funded_baseline = True`.

**Simulated Base Prompt:**
> "You are the lead portfolio asset manager. Review the `physical_risk_score` and `financial_exposure_score` for the entire portfolio. Your objective is to protect the highest value assets that are most exposed to risk. Calculate a `final_priority_score` by weighting the financial exposure at 60% and the physical risk at 40%. Rank the portfolio and approve the top 30% for mitigation funding."

---

## A Short Example

Imagine a retail property in Miami-Dade County:
* **Input Data:** `wind_risk_mph` = 175, `flood_zone` = VE, `noi` = $1,500,000, `insurance_premium` = $350,000

**Step 1: Risk Assessor**
* Wind Score: (175 - 100) / 100 = 0.75
* Flood Score: 1.0 (VE zone)
* `physical_risk_score`: (0.75 * 0.5) + (1.0 * 0.5) = **0.875**

**Step 2: Financial Analyst**
* NOI Score: 1,500,000 / 2,000,000 = 0.75
* Ins Score: 350,000 / 500,000 = 0.70
* `financial_exposure_score`: (0.75 * 0.6) + (0.70 * 0.4) = **0.730**

**Step 3: Prioritization Scorer**
* `final_priority_score`: (0.730 * 0.6) + (0.875 * 0.4) = **0.788**

Because this score is very high (driven by high value and high coastal risk), this property will easily rank in the top 30% and receive `funded_baseline = True`. This logic is exactly what causes the demographic disparity: wealthy coastal properties score highest financially and physically, leaving inland/lower-income properties unfunded.