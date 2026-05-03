# Deep Dive: The Agentic Scorer (`agentic_scorer.py`)

This document explains the mechanics of the baseline multi-agent system used to prioritize properties for hardening capital.

## Integration with Nvidia NIM

We use the `langchain_nvidia_ai_endpoints` library to connect to the **z-ai/glm4.7** model hosted on Nvidia NIM.

Because we process over 100 properties, we feed the data to the LLM in batches (size 20) and instruct it to return structured JSON arrays containing the `id` and the new computed scores. We maintain a deterministic fallback in case the LLM fails to format the JSON perfectly, ensuring the pipeline remains robust.

We use `LangGraph` to orchestrate the state graph and route the data through the three agents sequentially.

---

## 1. The Risk Assessor Agent

**What it does:** Evaluates the physical vulnerability of the property.
**Data Used:** `wind_risk_mph` (ASCE 7 proxy) and `flood_zone` (Real FEMA zone).

**The Base Prompt:**
> "You are an expert structural engineer and risk assessor. Review the provided FEMA 'flood_zone' (VE is high risk, AE is moderate, X is low) and ASCE 7 'wind_risk_mph' for the following commercial properties. Calculate a 'physical_risk_score' between 0.0 and 1.5, weighting wind and flood equally. High coastal hazard zones (VE) and winds over 150mph should approach the maximum score. Return a JSON array of objects..."

---

## 2. The Financial Analyst Agent

**What it does:** Evaluates the financial exposure and value at risk if the property is damaged.
**Data Used:** `noi` (Net Operating Income) and `insurance_premium`.

**The Base Prompt:**
> "You are a commercial real estate financial analyst. Review the 'noi' (Net Operating Income) and 'insurance_premium' for the following properties. Output a normalized 'financial_exposure_score' between 0.0 and 1.0. Weight the NOI heavily (60%) as it represents cash flow at risk, but also consider the insurance burden (40%). Assume max NOI is ~2,000,000 and max premium is ~500,000 for normalization. Return a JSON array of objects..."

---

## 3. The Prioritization Scorer Agent

**What it does:** The final decision node. It takes the output from the Risk Assessor and the Financial Analyst to create the final "fund-first" list.
**Data Used:** `physical_risk_score` and `financial_exposure_score`.

**The Base Prompt:**
> "You are the lead portfolio asset manager. Review the 'physical_risk_score' and 'financial_exposure_score'. Calculate a 'final_priority_score' by weighting the financial exposure at 60% and the physical risk at 40%. Return a JSON array of objects..."

*Note: After the LLM assigns the `final_priority_score` to all properties, we use `pandas` to strictly rank them and assign `funded_baseline = True` to the top 30%.*

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