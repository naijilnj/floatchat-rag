VIZ_SUGGESTION_PROMPT = """Based on the query results with columns:
{columns}

And data summary:
{summary}

Suggest the most appropriate visualization:
1. Chart type (time series, scatter, map, etc.)
2. X and Y axis variables
3. Additional dimensions (color, size, etc.)
4. Recommended aggregations
5. Interactive features to include"""