# ML System Design Case Studies

8 detailed ML system design write-ups covering the most commonly asked interview questions at FAANG companies (2025-2026).

Each case study follows a structured 6-step framework:
1. Problem Definition + Clarifying Questions
2. Metrics (Offline + Online)
3. High-Level Architecture (Mermaid diagrams)
4. Data Pipeline + Feature Engineering
5. Model Selection + Training Strategy
6. Serving, Monitoring, and Trade-offs

## Case Studies

| # | System | Key Concepts |
|---|--------|-------------|
| 1 | [Product Recommendation System](case-studies/01-recommendation-system.md) | Two-tower model, candidate generation + ranking, collaborative filtering |
| 2 | [Real-Time Fraud Detection](case-studies/02-fraud-detection.md) | Streaming pipeline, feature store, ensemble model, real-time serving |
| 3 | [Content Moderation System](case-studies/03-content-moderation.md) | Multimodal ML, human-in-the-loop, policy enforcement |
| 4 | [Search Ranking System](case-studies/04-search-ranking.md) | Learning-to-rank, BM25 + semantic search, query understanding |
| 5 | [News Feed Ranking](case-studies/05-newsfeed-ranking.md) | Multi-stage ranking, engagement prediction, diversity injection |
| 6 | [Ads CTR Prediction](case-studies/06-ads-ctr-prediction.md) | Deep + cross networks, feature interactions, calibration |
| 7 | [ETA Prediction (Maps)](case-studies/07-eta-prediction.md) | Geospatial features, graph neural networks, real-time traffic |
| 8 | [Model Monitoring & A/B Testing](case-studies/08-monitoring-ab-testing.md) | Data drift detection, experiment design, guardrail metrics |

## Framework Reference

Every case study answers these questions:
- What is the ML problem? (classification, regression, ranking?)
- What data do we need and how do we get it?
- What features matter and how do we compute them?
- What model architecture fits and why?
- How do we serve predictions at scale?
- How do we know the system is working?

## Author

Thejas Nagesh Gowda | MS in Artificial Intelligence, RIT
