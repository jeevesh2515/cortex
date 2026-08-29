---
title: Relational Database Indexing
tags: [sql, postgres, databases]
sensitivity: private
---

# Relational Database Indexing

Relational database management systems (RDBMS) like PostgreSQL use B-tree and LSM-tree index structures for fast primary key and range lookups.

SQL query optimization:
- Index scans vs sequential table scans.
- Query planner chooses access paths based on table statistics and cost estimation.
- Foreign key constraints maintain referential integrity across relational tables.
- This traditional tabular search paradigm differs fundamentally from embedding-based semantic retrieval.
