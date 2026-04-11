import json
import logging

from database.db import get_connection
from engine.portfolio import Portfolio

logger = logging.getLogger(__name__)

def save_portfolio(portfolio):
    logger.info(
        "save_portfolio — name: %s | tickers: %s | cash: %.2f",
        portfolio.name,
        portfolio.tickers,
        portfolio.initial_cash,
    )

    try:
        conn   = get_connection()
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
        logger.info("save_portfolio — success | name: %s", portfolio.name)

    except Exception as e:
        logger.error("save_portfolio — failed: %s", e, exc_info=True)
        raise

    finally:
        conn.close()

def list_portfolios():
    logger.debug("list_portfolios — querying DB")

    try:
        conn   = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
        SELECT id, name, start_date, end_date, interval
        FROM portfolios
        ORDER BY id DESC
        """)

        rows = cursor.fetchall()
        logger.debug("list_portfolios — found %d portfolio(s)", len(rows))
        return rows

    except Exception as e:
        logger.error("list_portfolios — failed: %s", e, exc_info=True)
        raise

    finally:
        conn.close()

def load_portfolio(portfolio_id):
    logger.info("load_portfolio — id: %s", portfolio_id)

    try:
        conn   = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
        SELECT name, initial_cash, commission_rate, slippage_rate,
               tickers, weights, strategy_map,
               start_date, end_date, interval
        FROM portfolios
        WHERE id = ?
        """, (portfolio_id,))

        row = cursor.fetchone()

        if row is None:
            logger.warning("load_portfolio — no portfolio found for id: %s", portfolio_id)
            return None

        portfolio = Portfolio(
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

        logger.info(
            "load_portfolio — success | name: %s | tickers: %s",
            portfolio.name,
            portfolio.tickers,
        )

        return portfolio

    except Exception as e:
        logger.error("load_portfolio — failed: %s", e, exc_info=True)
        raise

    finally:
        conn.close()

def update_portfolio(portfolio_id, portfolio):
    logger.info(
        "update_portfolio — id: %s | name: %s | tickers: %s",
        portfolio_id,
        portfolio.name,
        portfolio.tickers,
    )

    try:
        conn   = get_connection()
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

        logger.info(
            "update_portfolio — success | id: %s | name: %s",
            portfolio_id,
            portfolio.name,
        )

    except Exception as e:
        logger.error(
            "update_portfolio — failed | id: %s | error: %s",
            portfolio_id,
            e,
            exc_info=True,
        )
        raise

    finally:
        conn.close()

def delete_portfolio(portfolio_id):
    logger.info("delete_portfolio — id: %s", portfolio_id)

    try:
        conn   = get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM portfolios WHERE id = ?", (portfolio_id,))
        conn.commit()

        logger.info("delete_portfolio — success | id: %s", portfolio_id)

    except Exception as e:
        logger.error(
            "delete_portfolio — failed | id: %s | error: %s",
            portfolio_id,
            e,
            exc_info=True,
        )
        raise

    finally:
        conn.close()