import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
import sqlite3
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
import pandas as pd

class SQLiteSpanExporter(SpanExporter):
    def __init__(self, db_path="traces.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                name TEXT,
                start_time INTEGER,
                end_time INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL
            )
        """)
        self.conn.execute("DELETE FROM spans") # clean up for fresh run
        self.conn.commit()

    def export(self, spans):
        for span in spans:
            attrs = dict(span.attributes or {})
            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    span.name,
                    span.start_time,
                    span.end_time,
                    attrs.get("input_tokens"),
                    attrs.get("output_tokens"),
                    attrs.get("cost"),
                ),
            )
        self.conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.conn.close()

    def force_flush(self):
        return True

provider = TracerProvider()
# Add Console Exporter for Q1, Q3
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
# Add SQLite Exporter for Q4, Q5, Q6
provider.add_span_processor(SimpleSpanProcessor(SQLiteSpanExporter("traces.db")))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("llm-zoomcamp")

from starter import index, client
from rag_helper import RAGBase

class RAGTraced(RAGBase):
    def search(self, query, num_results=5):
        with tracer.start_as_current_span("search") as span:
            return super().search(query, num_results)
            
    def llm(self, prompt):
        with tracer.start_as_current_span("llm") as span:
            response = super().llm(prompt)
            # Q2: metrics
            if hasattr(response, 'usage'):
                usage = response.usage
                span.set_attribute("input_tokens", usage.input_tokens)
                span.set_attribute("output_tokens", usage.output_tokens)
                # dummy cost
                cost = (usage.input_tokens * 0.15 + usage.output_tokens * 0.6) / 1000000
                span.set_attribute("cost", cost)
            return response
            
    def rag(self, query):
        with tracer.start_as_current_span("rag") as span:
            return super().rag(query)

rag_traced = RAGTraced(index=index, llm_client=client)

query = "How does the agentic loop keep calling the model until it stops?"

print("--- RUN 1 ---")
rag_traced.rag(query)
print("--- RUN 2 ---")
rag_traced.rag(query)
print("--- RUN 3 ---")
rag_traced.rag(query)
print("--- RUN 4 ---")
rag_traced.rag(query)

print("--- DB STATS ---")
conn = sqlite3.connect("traces.db")
df = pd.read_sql_query("SELECT * FROM spans", conn)
print(df)
df['duration_ms'] = (df['end_time'] - df['start_time']) / 1e6
print("\nDurations:")
print(df.groupby('name')['duration_ms'].sum())
print("\nTokens:")
print(df[df['name'] == 'llm']['input_tokens'])
