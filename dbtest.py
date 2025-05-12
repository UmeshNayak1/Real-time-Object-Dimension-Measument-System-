import psycopg2

# DB_SETTINGS = {
#     "host": "localhost",
#     "port": 5432,
#     "database": " finalyearproject",
#     "user": "postgres",
#     "password": "admin"
# }
DB_SETTINGS = {
    "host": "dpg-d0h37ejuibrs7385ja8g-a",
    "port": 5432,
    "database": "object_dimension_measure",
    "user": "object_dimension_measure_user",
    "password": "JPIIU4BbMIiDDlGszmQFhGqhFPsZT99g"
}

try:
    conn = psycopg2.connect(**DB_SETTINGS)
    print("✅ Database connection successful!")
    conn.close()
except Exception as e:
    print(f"❌ Database connection failed: {e}")
