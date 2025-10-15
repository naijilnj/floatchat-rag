import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from datetime import datetime
import os

# Database connection parameters
DB_PARAMS = {
    'host': 'localhost',
    'database': 'argo_db',
    'user': 'postgres',  # Replace with your PostgreSQL username
    'password': '142003'  # Replace with your PostgreSQL password
}

def create_table():
    """Create the table if it doesn't exist"""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS buoy_measurements (
        id SERIAL PRIMARY KEY,
        buoy_id VARCHAR(10),
        parameter VARCHAR(50),
        timestamp TIMESTAMP,
        value FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_buoy_timestamp 
        ON buoy_measurements(buoy_id, timestamp);
    """
    
    try:
        cur.execute(create_table_query)
        conn.commit()
        print("Table created successfully")
    except Exception as e:
        print(f"Error creating table: {str(e)}")
    finally:
        cur.close()
        conn.close()

def load_data_to_db(csv_file):
    """Load data from CSV to PostgreSQL"""
    try:
        # Read CSV file
        df = pd.read_csv(csv_file,nrows=200000)
        
        # Convert timestamp string to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create SQLAlchemy engine
        engine = create_engine(
            f'postgresql://{DB_PARAMS["user"]}:{DB_PARAMS["password"]}@'
            f'{DB_PARAMS["host"]}/{DB_PARAMS["database"]}'
        )
        
        # Load data to database
        df.to_sql(
            'buoy_measurements', 
            engine, 
            if_exists='append', 
            index=False,
            method='multi',
            chunksize=1000
        )
        
        print(f"Successfully loaded {len(df)} records to database")
        
        # Print summary
        print("\nData Summary:")
        print(f"Records per buoy:\n{df['buoy_id'].value_counts()}")
        print(f"\nRecords per parameter:\n{df['parameter'].value_counts()}")
        
    except Exception as e:
        print(f"Error loading data to database: {str(e)}")

def main():
    # Create table
    create_table()
    
    # Find the most recent CSV file in the current directory
    csv_files = [f for f in os.listdir('.') if f.startswith('buoy_data_') and f.endswith('.csv')]
    if not csv_files:
        print("No buoy data CSV files found!")
        return
    
    latest_csv = max(csv_files)
    print(f"Loading data from {latest_csv}")
    
    # Load data
    load_data_to_db(latest_csv)

if __name__ == "__main__":
    main()