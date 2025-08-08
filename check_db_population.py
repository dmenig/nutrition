import os
import psycopg2
from app.core.config import settings

# Use DATABASE_URL from environment variables if available, otherwise use the hardcoded one
database_url = os.getenv("DATABASE_URL", settings.DATABASE_URL)

# Connect to the database
conn = psycopg2.connect(database_url)
cur = conn.cursor()

# Execute query to count rows in foods table
cur.execute("SELECT COUNT(*) FROM foods;")
foods_count = cur.fetchone()[0]
print(f"Number of rows in foods table: {foods_count}")

# Execute query to count rows in food_logs table
cur.execute("SELECT COUNT(*) FROM food_logs;")
food_logs_count = cur.fetchone()[0]
print(f"Number of rows in food_logs table: {food_logs_count}")

# Close connections
cur.close()
conn.close()
