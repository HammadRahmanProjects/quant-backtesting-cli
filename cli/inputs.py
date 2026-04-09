from datetime import datetime
from typing import List
import questionary

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)

from data.fetchers import validate_ticker

console = Console()

def get_valid_date(prompt: str) -> datetime:
    while True:
        date_str = input(prompt).strip()

        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            print("Invalid date. Please use YYYY-MM-DD format.")

def confirm_action(message: str = "Are you sure?") -> bool:
    return questionary.confirm(
        message,
        default=False
    ).ask()

def confirm_action(message: str = "Are you sure?") -> bool:
    return questionary.confirm(
        message,
        default=False
    ).ask() or False

console = Console()

def prompt_text(message: str):
    answer = questionary.text(message).ask()
    return answer

def prompt_select(message: str, choices):
    """
    Returns:
        selected choice -> normal selection
        None            -> user pressed Esc / canceled
    """
    return questionary.select(
        message,
        choices=choices,
        pointer="➤",
    ).ask()

def prompt_confirm(message: str, default: bool = False):
    """
    Returns:
        bool -> yes / no
        None -> Esc / canceled
    """
    return questionary.confirm(
        message,
        default=default,
    ).ask()

def get_valid_tickers() -> List[str]:
    entered_tickers = input(
        "Enter tickers (comma separated, e.g. AAPL,MSFT,TSLA): "
    ).strip()

    tickers = [ticker.strip().upper() for ticker in entered_tickers.split(",") if ticker.strip()]

    while True:
        if len(tickers) == 0:
            print("Please enter at least one ticker.")
            entered_tickers = input(
                "Enter tickers (comma separated, e.g. AAPL,MSFT,TSLA): "
            ).strip()
            tickers = [ticker.strip().upper() for ticker in entered_tickers.split(",") if ticker.strip()]
            continue

        if len(set(tickers)) != len(tickers):
            print("Duplicate tickers are not allowed.")
            entered_tickers = input(
                "Re-enter tickers (comma separated): "
            ).strip()
            tickers = [ticker.strip().upper() for ticker in entered_tickers.split(",") if ticker.strip()]
            continue

        invalid_tickers = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Validating tickers...", total=len(tickers))

            for ticker in tickers:
                if not validate_ticker(ticker):
                    invalid_tickers.append(ticker)
                progress.advance(task)

        if not invalid_tickers:
            console.print("[#26A69A]All tickers validated successfully.[/#26A69A]")
            return tickers

        console.print(
            Panel(
                f"Invalid or unavailable tickers:\n[#EF5350]{', '.join(invalid_tickers)}[#EF5350]",
                title="Ticker Validation Failed",
                border_style="red",
            )
        )

        replacement_input = input(
            f"Re-enter only these tickers ({', '.join(invalid_tickers)}), comma separated: "
        ).strip()

        replacement_tickers = [
            ticker.strip().upper()
            for ticker in replacement_input.split(",")
            if ticker.strip()
        ]

        if len(replacement_tickers) != len(invalid_tickers):
            print("You must re-enter the same number of invalid tickers.")
            continue

        replacement_index = 0
        updated_tickers = []

        for ticker in tickers:
            if ticker in invalid_tickers:
                updated_tickers.append(replacement_tickers[replacement_index])
                replacement_index += 1
            else:
                updated_tickers.append(ticker)

        tickers = updated_tickers