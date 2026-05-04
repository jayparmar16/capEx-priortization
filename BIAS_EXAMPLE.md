# Algorithmic Bias in Priority Ranking: A Concrete Example

When evaluating algorithmic fairness in capital allocation, it is easy to assume that because an algorithm uses "objective" inputs (like Property Value, Net Operating Income, and Weather Risk), the output must be fair.

However, because financial metrics (Property Value) are intrinsically tied to neighborhood demographics (Income Quartile), an algorithm that optimizes for financial exposure will inevitably prioritize high-income neighborhoods.

We searched the results of our **Live LLM Agentic Scorer** (`moonshotai/kimi-k2-thinking`) to find a concrete example demonstrating exactly why a Parity Re-ranker is necessary.

### The Bias Example

Here are two properties from our simulated Florida portfolio. The LLM algorithm chose to fund the High-Income property and ignore the Low-Income property.

| Metric | High-Income Property (Funded) | Low-Income Property (Unfunded) |
| :--- | :--- | :--- |
| **Location** | Miami-Dade County | Miami-Dade County |
| **Neighborhood Income** | $158,984 (Quartile 4) | $72,258 (Quartile 2) |
| **Flood Zone (FEMA)** | X (Minimal) | X (Minimal) |
| **Wind Risk (ASCE)** | 165.1 mph | **170.7 mph** (Higher Risk) |
| **Physical Risk Score** | 0.376 | **0.404** (Higher Risk) |
| **Property Value** | $18,315,987 | $7,203,968 |
| **Financial Exposure Score** | 0.483 | 0.197 |
| **Final Priority Score** | **0.440** (Ranked #1) | 0.280 (Ranked #88) |

### Analysis: Why did this happen?

The Low-Income property actually faces a **higher physical threat** from hurricanes (170.7 mph wind risk compared to 165.1 mph for the High-Income property). If the goal of the mitigation capital was purely to harden the most physically vulnerable buildings, the Low-Income property should have won.

However, the algorithm is instructed to weight *Financial Exposure* (NOI and Property Value) at 60%, and *Physical Risk* at 40%. Because the High-Income property is worth $18.3M (driven by the high median income of its surrounding Census tract), its Financial Exposure score completely overwhelmed its lower physical risk.

**This is the definition of systemic algorithmic bias in real estate.** The system does not explicitly look at race or income, but by optimizing for "value at risk," it systematically diverts life-saving structural hardening capital away from vulnerable working-class neighborhoods and channels it toward wealthier areas.

This is exactly why the **Parity Re-ranker** is required to correct the final list.