import os
import dlt
from dlt.sources.rest_api import RESTAPIConfig, rest_api_resources
from dotenv import load_dotenv

load_dotenv()

@dlt.source
def logfire_source():
    # Use the token from .env
    read_token = os.getenv("LOGFIRE_TOKEN")
    
    config: RESTAPIConfig = {
        "client": {
            "base_url": "https://logfire-us.pydantic.dev/v1/",
            "auth": {
                "type": "bearer",
                "token": read_token,
            },
        },
        "resources": [
            {
                "name": "query",
                "endpoint": {
                    "path": "query",
                    "params": {
                        "sql": "SELECT * FROM records"
                    }
                }
            }
        ],
    }
    
    yield from rest_api_resources(config)

def load_data():
    pipeline = dlt.pipeline(
        pipeline_name='logfire_pipeline',
        destination='duckdb',
        dataset_name='agent_traces',
    )
    
    load_info = pipeline.run(logfire_source())
    print(load_info)

if __name__ == "__main__":
    load_data()
