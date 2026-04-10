import json

from database.db import get_connection
from engine.portfolio import Portfolio

def save_portfolio(portfolio):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO portfolios (
        name, initial_cash, commission_rate, slippage_rate,
        tickers, weights, strategy_map,
        start_date, end_date, interval
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        portfolio.name,
        portfolio.initial_cash,
        portfolio.commission_rate,
        portfolio.slippage_rate,
        json.dumps(portfolio.tickers),
        json.dumps(portfolio.weights),
        json.dumps(portfolio.strategy_map),
        portfolio.start_date,
        portfolio.end_date,
        portfolio.interval,
    ))

    conn.commit()
    conn.close()

def list_portfolios():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, name, start_date, end_date, interval
    FROM portfolios
    ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows

def load_portfolio(portfolio_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT name, initial_cash, commission_rate, slippage_rate,
           tickers, weights, strategy_map,
           start_date, end_date, interval
    FROM portfolios
    WHERE id = ?
    """, (portfolio_id,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return Portfolio(
        name=row[0],
        initial_cash=row[1],
        commission_rate=row[2],
        slippage_rate=row[3],
        tickers=json.loads(row[4]),
        weights=json.loads(row[5]),
        strategy_map=json.loads(row[6]),
        start_date=row[7],
        end_date=row[8],
        interval=row[9],
    )

def update_portfolio(portfolio_id, portfolio):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE portfolios SET
        name = ?,
        initial_cash = ?,
        commission_rate = ?,
        slippage_rate = ?,
        tickers = ?,
        weights = ?,
        strategy_map = ?,
        start_date = ?,
        end_date = ?,
        interval = ?
    WHERE id = ?
    """, (
        portfolio.name,
        portfolio.initial_cash,
        portfolio.commission_rate,
        portfolio.slippage_rate,
        json.dumps(portfolio.tickers),
        json.dumps(portfolio.weights),
        json.dumps(portfolio.strategy_map),
        portfolio.start_date,
        portfolio.end_date,
        portfolio.interval,
        portfolio_id,
    ))

    conn.commit()
    conn.close()

def delete_portfolio(portfolio_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM portfolios WHERE id = ?", (portfolio_id,))

    conn.commit()
    conn.close()