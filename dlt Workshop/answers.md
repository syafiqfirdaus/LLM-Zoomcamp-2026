# dlt Workshop Homework Answers

## Question 1: Instrument the agent with Logfire
**Question:** For the query "How do I run Ollama locally?", how many spans does a single agent run produce?
**Answer:** `5`

## Question 2: Load traces into DuckDB with dlt
**Question:** How many tables did dlt create?
**Answer:** `24`

## Question 3: Query traces with an agent
**Question:** Find the input token usage for the agent run from Q1. Sum them across all LLM calls within the trace. The number depends on how many searches the agent made, so report the range it falls into.
**Answer:** `1500 - 5000`
