from typing import List

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from data.fetchers import validate_ticker

console = Console()


def prompt_text(message: str):
    return questionary.text(message).ask()


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


def confirm_action(message: str = "Are you sure?") -> bool:
    result = questionary.confirm(
        message,
        default=False,
    ).ask()
    return result or False


def get_valid_date(prompt: str):
    """
    Returns:
        datetime -> valid date entered
        None     -> user pressed Esc / canceled
    """
    from datetime import datetime

    while True:
        date_str = prompt_text(prompt)

        if date_str is None:
            return None

        date_str = date_str.strip()

        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            console.print("[red]Invalid date. Please use YYYY-MM-DD format.[/red]")


def get_valid_tickers() -> List[str]:
    """
    Returns:
        List[str] -> validated list of tickers
        None      -> user pressed Esc / canceled
    """
    raw = prompt_text("Enter tickers (comma separated, e.g. AAPL,MSFT,TSLA):")

    if raw is None:
        return None

    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]

    while True:
        if len(tickers) == 0:
            raw = prompt_text("Please enter at least one ticker:")
            if raw is None:
                return None
            tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
            continue

        if len(set(tickers)) != len(tickers):
            console.print("[red]Duplicate tickers are not allowed.[/red]")
            raw = prompt_text("Re-enter tickers (comma separated):")
            if raw is None:
                return None
            tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
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
                f"Invalid or unavailable tickers:\n[#EF5350]{', '.join(invalid_tickers)}[/#EF5350]",
                title="Ticker Validation Failed",
                border_style="red",
            )
        )

        raw = prompt_text(
            f"Re-enter only these tickers ({', '.join(invalid_tickers)}), comma separated:"
        )

        if raw is None:
            return None

        replacement_tickers = [
            t.strip().upper() for t in raw.split(",") if t.strip()
        ]

        if len(replacement_tickers) != len(invalid_tickers):
            console.print(
                f"[red]You must re-enter exactly {len(invalid_tickers)} ticker(s).[/red]"
            )
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