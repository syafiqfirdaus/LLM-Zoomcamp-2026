import duckdb

con = duckdb.connect('logfire_pipeline.duckdb')

# Q2: How many tables did dlt create?
res = con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'agent_traces'").fetchone()
print("Q2 (Table Count):", res[0])

# Let's inspect the columns in the main table to find where tokens are
tables = con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'agent_traces'").fetchall()
print("\nTables:")
for t in tables:
    print(t[0])

# Get columns for query__values or equivalent
main_table = [t[0] for t in tables if t[0] == 'query'][0] if 'query' in [t[0] for t in tables] else tables[0][0]
cols = con.execute(f"DESCRIBE agent_traces.{main_table}").fetchall()
print(f"\nColumns in {main_table}:")
for c in cols:
    if 'token' in c[0].lower():
        print("MATCH:", c[0])
    elif 'usage' in c[0].lower():
        print("MATCH:", c[0])

# Sum of tokens
res_tokens = con.execute("SELECT SUM(gen_ai_usage_input_tokens) FROM agent_traces.query__values").fetchone()
print("Q3 (Token Sum):", res_tokens[0])

res_tokens2 = con.execute("SELECT SUM(gen_ai_aggregated_usage_input_tokens) FROM agent_traces.query__values").fetchone()
print("Q3 (Token Sum Aggregated):", res_tokens2[0] if res_tokens2 else None)



