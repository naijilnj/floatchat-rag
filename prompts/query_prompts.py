SQL_TEMPLATE = """Given the following database schema:
{schema}

User Question: {question}

Generate a SQL query that:
1. Precisely addresses the user's question
2. Uses appropriate time buckets for time series analysis
3. Includes necessary joins and conditions
4. Limits results to manageable sizes (max 1000 rows)
5. Handles missing or null values appropriately

Additional Requirements:
- Use TimescaleDB time_bucket() function for time series
- Include parameter-specific filters
- Consider depth values when relevant
- Add appropriate ordering

Output Format:
1. SQL Query
2. Brief explanation of the query
3. Suggested visualization type"""

QUERY_REFINEMENT_PROMPT = """Based on the user's follow-up question:
{follow_up}

Previous context:
{context}

Modify the previous query to:
1. Incorporate new constraints
2. Maintain existing relevant filters
3. Update visualization suggestions if needed"""