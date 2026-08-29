---
title: Thermal Management
tags: [hardware, apple-silicon, thermal]
sensitivity: private
---

# Thermal Management

Fanless devices such as the MacBook Air lack active cooling and experience thermal saturation under sustained CPU or GPU workloads.
When chassis temperature rises, macOS throttles processor clock speeds (CPU Speed Limit drops below 100%).

Cortex Thermal Policy:
- Background indexing yields to interactive queries.
- When on battery or when thermal state is throttled, backfill concurrency is reduced or suspended.
- System probes monitor `pmset` to observe power source, battery percentage, load average, and speed limit.
- Ensures local RAG indexing runs quietly in the background without making the laptop hot or unresponsive.
- Linked to [[Retrieval Systems]] performance tuning.
