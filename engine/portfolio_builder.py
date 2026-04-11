from datetime import datetime

import questionary
from rich.console import Console

from strategies.registry import AVAILABLE_STRATEGIES
from engine.portfolio import Portfolio
from cli.inputs import (
    get_valid_date,
    get_valid_tickers,
    prompt_text,
)

console = Console()


def _prompt_positive_float(prompt_message: str):
    while True:
        raw_value = prompt_text(prompt_message)
        if raw_value is None:
            return None

        try:
            value = float(raw_value)
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            console.print("[red]Invalid input. Enter a positive number.[/red]")


def _prompt_non_negative_float(prompt_message: str):
    while True:
        raw_value = prompt_text(prompt_message)
        if raw_value is None:
            return None

        try:
            value = float(raw_value)
            if value < 0:
                raise ValueError
            return value
        except ValueError:
            console.print("[red]Invalid input. Enter a non-negative number.[/red]")


def _prompt_weights(tickers):
    while True:
        weight_input = prompt_text("Enter weights (comma separated, e.g. 0.5,0.3,0.2):")
        if weight_input is None:
            return None

        try:
            weights = [
                float(w.strip())
                for w in weight_input.split(",")
                if w.strip()
            ]

            if len(weights) != len(tickers):
                console.print("[red]Number of weights must match number of tickers.[/red]")
                continue

            if any(w < 0 for w in weights):
                console.print("[red]Weights must be non-negative.[/red]")
                continue

            if abs(sum(weights) - 1.0) > 1e-6:
                console.print("[red]Weights must sum to 1.0.[/red]")
                continue

            return weights

        except ValueError:
            console.print("[red]Invalid weights. Enter numeric values only.[/red]")


def _prompt_strategy_map(tickers):
    """
    For each ticker, present a scrollable questionary.select menu of
    available strategies. No typing required — no typo risk.
    Returns None if the user cancels at any point.
    """
    available_strategies = list(AVAILABLE_STRATEGIES.keys())
    strategy_map = {}

    for ticker in tickers:
        # Build choices with parameter info shown as a hint
        choices = []
        for strategy_name in available_strategies:
            strategy_class = AVAILABLE_STRATEGIES[strategy_name]
            param_names = strategy_class.get_param_names()
            params_text = ", ".join(param_names)
            choices.append(
                questionary.Choice(
                    title=f"{strategy_name}  ({params_text})",
                    value=strategy_name,
                )
            )

        selected = questionary.select(
            f"Select strategy for {ticker}:",
            choices=choices,
            pointer="➤",
        ).ask()

        if selected is None:
            return None

        strategy_map[ticker] = {"name": selected}

    return strategy_map


def _prompt_dates():
    while True:
        start_dt = get_valid_date("Enter start date (YYYY-MM-DD):")
        if start_dt is None:
            return None, None

        end_dt = get_valid_date("Enter end date (YYYY-MM-DD):")
        if end_dt is None:
            return None, None

        today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

        if start_dt > today:
            console.print("[red]Start date cannot be in the future.[/red]")
            continue

        if end_dt > today:
            console.print("[red]End date cannot be in the future.[/red]")
            continue

        if end_dt <= start_dt:
            console.print("[red]End date must be AFTER start date.[/red]")
            continue

        return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def _format_strategy_map(strategy_map):
    formatted = []
    for ticker, strategy_info in strategy_map.items():
        formatted.append(f"{ticker}: {strategy_info['name']}")
    return " | ".join(formatted)


def _print_portfolio_review(data):
    console.print("\n[#2962FF]--- Review Portfolio ---[/#2962FF]")
    console.print(f"[bold]Name:[/bold]            {data['name']}")
    console.print(f"[bold]Initial Cash:[/bold]    {data['initial_cash']}")
    console.print(f"[bold]Commission Rate:[/bold] {data['commission_rate']}")
    console.print(f"[bold]Slippage Rate:[/bold]   {data['slippage_rate']}")
    console.print(f"[bold]Tickers:[/bold]         {data['tickers']}")
    console.print(f"[bold]Weights:[/bold]         {data['weights']}")
    console.print(f"[bold]Strategies:[/bold]      {_format_strategy_map(data['strategy_map'])}")
    console.print(f"[bold]Start Date:[/bold]      {data['start_date']}")
    console.print(f"[bold]End Date:[/bold]        {data['end_date']}")
    console.print(f"[bold]Interval:[/bold]        {data['interval']}")


def _edit_name(data):
    value = prompt_text("Enter portfolio name:")
    if value is not None:
        data["name"] = value.strip()


def _edit_initial_cash(data):
    value = _prompt_positive_float("Enter initial cash:")
    if value is not None:
        data["initial_cash"] = value


def _edit_commission_rate(data):
    value = _prompt_non_negative_float("Enter commission rate (e.g. 0.001):")
    if value is not None:
        data["commission_rate"] = value


def _edit_slippage_rate(data):
    value = _prompt_non_negative_float("Enter slippage rate (e.g. 0.001):")
    if value is not None:
        data["slippage_rate"] = value


def _edit_tickers(data):
    tickers = get_valid_tickers()
    if tickers is None:
        return

    weights = _prompt_weights(tickers)
    if weights is None:
        return

    strategy_map = _prompt_strategy_map(tickers)
    if strategy_map is None:
        return

    data["tickers"] = tickers
    data["weights"] = weights
    data["strategy_map"] = strategy_map


def _edit_weights(data):
    weights = _prompt_weights(data["tickers"])
    if weights is not None:
        data["weights"] = weights


def _edit_strategies(data):
    strategy_map = _prompt_strategy_map(data["tickers"])
    if strategy_map is not None:
        data["strategy_map"] = strategy_map


def _edit_dates(data):
    start_date, end_date = _prompt_dates()
    if start_date is not None and end_date is not None:
        data["start_date"] = start_date
        data["end_date"] = end_date


def _edit_interval(data):
    value = prompt_text("Enter interval (e.g. 1d, 1h, 1m):")
    if value is not None:
        data["interval"] = value.strip()


EDIT_ACTIONS = {
    "Edit Name":            _edit_name,
    "Edit Initial Cash":    _edit_initial_cash,
    "Edit Commission Rate": _edit_commission_rate,
    "Edit Slippage Rate":   _edit_slippage_rate,
    "Edit Tickers":         _edit_tickers,
    "Edit Weights":         _edit_weights,
    "Edit Strategies":      _edit_strategies,
    "Edit Dates":           _edit_dates,
    "Edit Interval":        _edit_interval,
}


def _review_and_edit_portfolio_data(data):
    while True:
        _print_portfolio_review(data)

        choice = questionary.select(
            "Review portfolio and choose what to do:",
            choices=[
                "Edit Name",
                "Edit Initial Cash",
                "Edit Commission Rate",
                "Edit Slippage Rate",
                "Edit Tickers",
                "Edit Weights",
                "Edit Strategies",
                "Edit Dates",
                "Edit Interval",
                questionary.Separator(),
                "Confirm Portfolio",
                "Cancel",
            ],
            pointer="➤",
        ).ask()

        if choice is None or choice == "Cancel":
            return None

        if choice == "Confirm Portfolio":
            return data

        action = EDIT_ACTIONS.get(choice)
        if action:
            action(data)


def create_portfolio():
    console.print("\n[#2962FF]--- Create New Portfolio ---[/#2962FF]")
    console.print("[#9E9E9E]Press Esc at any prompt to return to the main menu.[/#9E9E9E]")

    name = prompt_text("Enter portfolio name:")
    if name is None:
        return None

    initial_cash = _prompt_positive_float("Enter initial cash:")
    if initial_cash is None:
        return None

    commission_rate = _prompt_non_negative_float("Enter commission rate (e.g. 0.001):")
    if commission_rate is None:
        return None

    slippage_rate = _prompt_non_negative_float("Enter slippage rate (e.g. 0.001):")
    if slippage_rate is None:
        return None

    tickers = get_valid_tickers()
    if tickers is None:
        return None

    weights = _prompt_weights(tickers)
    if weights is None:
        return None

    strategy_map = _prompt_strategy_map(tickers)
    if strategy_map is None:
        return None

    start_date, end_date = _prompt_dates()
    if start_date is None or end_date is None:
        return None

    interval = prompt_text("Enter interval (e.g. 1d, 1h, 1m):")
    if interval is None:
        return None

    portfolio_data = {
        "name":            name.strip(),
        "initial_cash":    initial_cash,
        "commission_rate": commission_rate,
        "slippage_rate":   slippage_rate,
        "tickers":         tickers,
        "weights":         weights,
        "strategy_map":    strategy_map,
        "start_date":      start_date,
        "end_date":        end_date,
        "interval":        interval.strip(),
    }

    reviewed_data = _review_and_edit_portfolio_data(portfolio_data)
    if reviewed_data is None:
        return None

    portfolio = Portfolio(
        name=reviewed_data["name"],
        initial_cash=reviewed_data["initial_cash"],
        commission_rate=reviewed_data["commission_rate"],
        slippage_rate=reviewed_data["slippage_rate"],
        tickers=reviewed_data["tickers"],
        weights=reviewed_data["weights"],
        strategy_map=reviewed_data["strategy_map"],
        start_date=reviewed_data["start_date"],
        end_date=reviewed_data["end_date"],
        interval=reviewed_data["interval"],
    )

    console.print("\n[#26A69A]Portfolio created successfully.[/#26A69A]")
    console.print(f"[#9E9E9E]{portfolio}[/#9E9E9E]")

    return portfolio


def get_allocated_cash(portfolio, ticker):
    index = portfolio.tickers.index(ticker)
    weight = portfolio.weights[index]
    return portfolio.initial_cash * weight


def edit_portfolio(portfolio):
    """Load an existing portfolio into the review/edit flow and return the updated version."""
    portfolio_data = {
        "name":            portfolio.name,
        "initial_cash":    portfolio.initial_cash,
        "commission_rate": portfolio.commission_rate,
        "slippage_rate":   portfolio.slippage_rate,
        "tickers":         portfolio.tickers,
        "weights":         portfolio.weights,
        "strategy_map":    portfolio.strategy_map,
        "start_date":      portfolio.start_date,
        "end_date":        portfolio.end_date,
        "interval":        portfolio.interval,
    }

    reviewed_data = _review_and_edit_portfolio_data(portfolio_data)
    if reviewed_data is None:
        return None

    return Portfolio(
        name=reviewed_data["name"],
        initial_cash=reviewed_data["initial_cash"],
        commission_rate=reviewed_data["commission_rate"],
        slippage_rate=reviewed_data["slippage_rate"],
        tickers=reviewed_data["tickers"],
        weights=reviewed_data["weights"],
        strategy_map=reviewed_data["strategy_map"],
        start_date=reviewed_data["start_date"],
        end_date=reviewed_data["end_date"],
        interval=reviewed_data["interval"],
    )