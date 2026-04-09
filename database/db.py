import sqlite3

DB_NAME = "quant.db"

def get_connection():
    return sqlite3.connect(DB_NAME)

def initialize_database():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS portfolios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        initial_cash REAL NOT NULL,
        commission_rate REAL NOT NULL,
        slippage_rate REAL NOT NULL,
        tickers TEXT NOT NULL,
        weights TEXT NOT NULL,
        strategy_map TEXT NOT NULL,
        start_date TEXT NOT NULL,
        end_date TEXT NOT NULL,
        interval TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()