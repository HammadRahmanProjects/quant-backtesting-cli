from datetime import datetime

from rich.console import Console

from strategies.registry import AVAILABLE_STRATEGIES
from engine.portfolio import Portfolio
from cli.inputs import (
    get_valid_date,
    get_valid_tickers,
    prompt_text,
)

console = Console()

def _parse_param_value(raw_value, default_value):
    if isinstance(default_value, bool):
        lowered = raw_value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
        raise ValueError("Enter a valid boolean value.")

    if isinstance(default_value, int):
        return int(raw_value)

    if isinstance(default_value, float):
        return float(raw_value)

    return raw_value

def prompt_strategy_params(strategy_class):
    param_names = strategy_class.get_param_names()
    default_params = strategy_class.get_default_params()

    strategy_params = {}

    for param_name in param_names:
        default_value = default_params[param_name]

        while True:
            raw_value = prompt_text(f"Enter {param_name} [{default_value}]:")
            if raw_value is None:
                return None

            raw_value = raw_value.strip()

            if raw_value == "":
                strategy_params[param_name] = default_value
                break

            try:
                strategy_params[param_name] = _parse_param_value(raw_value, default_value)
                break
            except ValueError:
                console.print(f"[red]Invalid value for {param_name}.[/red]")

    return strategy_params

def create_portfolio():
    console.print("\n[#2962FF]--- Create New Portfolio ---[/#2962FF]")
    console.print("[#9E9E9E]Press Esc at any prompt to return to the main menu.[/#9E9E9E]")

    name = prompt_text("Enter portfolio name:")
    if name is None:
        return None

    while True:
        raw_initial_cash = prompt_text("Enter initial cash:")
        if raw_initial_cash is None:
            return None

        try:
            initial_cash = float(raw_initial_cash)
            if initial_cash <= 0:
                raise ValueError
            break
        except ValueError:
            console.print("[red]Invalid input. Enter a positive number.[/red]")

    while True:
        raw_commission_rate = prompt_text("Enter commission rate (e.g. 0.001):")
        if raw_commission_rate is None:
            return None

        try:
            commission_rate = float(raw_commission_rate)
            if commission_rate < 0:
                raise ValueError
            break
        except ValueError:
            console.print("[red]Invalid input. Enter a non-negative number.[/red]")

    while True:
        raw_slippage_rate = prompt_text("Enter slippage rate (e.g. 0.001):")
        if raw_slippage_rate is None:
            return None

        try:
            slippage_rate = float(raw_slippage_rate)
            if slippage_rate < 0:
                raise ValueError
            break
        except ValueError:
            console.print("[red]Invalid input. Enter a non-negative number.[/red]")

    tickers = get_valid_tickers()
    if tickers is None:
        return None

    while True:
        weight_input = prompt_text("Enter weights (comma separated, e.g. 0.5,0.3,0.2):")
        if weight_input is None:
            return None

        try:
            weights = [float(weight.strip()) for weight in weight_input.split(",") if weight.strip()]

            if len(weights) != len(tickers):
                console.print("[red]Number of weights must match number of tickers.[/red]")
                continue

            if any(weight < 0 for weight in weights):
                console.print("[red]Weights must be non-negative.[/red]")
                continue

            if abs(sum(weights) - 1.0) > 1e-6:
                console.print("[red]Weights must sum to 1.0.[/red]")
                continue

            break

        except ValueError:
            console.print("[red]Invalid weights. Enter numeric values only.[/red]")

    available_strategies = list(AVAILABLE_STRATEGIES.keys())
    strategy_map = {}

    console.print("\n[bold]Available strategies:[/bold]")
    for strategy in available_strategies:
        strategy_class = AVAILABLE_STRATEGIES[strategy]
        param_names = strategy_class.get_param_names()
        params_text = ", ".join(param_names)
        console.print(f"- {strategy}({params_text})")

    for ticker in tickers:
        while True:
            strategy_name = prompt_text(f"Select strategy for {ticker}:")
            if strategy_name is None:
                return None

            strategy_name = strategy_name.strip().lower()

            if strategy_name not in available_strategies:
                console.print("[red]Invalid strategy name. Please choose from the list above.[/red]")
                continue

            strategy_class = AVAILABLE_STRATEGIES[strategy_name]
            strategy_params = prompt_strategy_params(strategy_class)

            if strategy_params is None:
                return None

            strategy_map[ticker] = {
                "name": strategy_name,
                "params": strategy_params,
            }
            break

    while True:
        start_dt = get_valid_date("Enter start date (YYYY-MM-DD): ")
        if start_dt is None:
            return None

        end_dt = get_valid_date("Enter end date (YYYY-MM-DD): ")
        if end_dt is None:
            return None

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

        break

    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")

    interval = prompt_text("Enter interval (e.g. 1d, 1h, 1m):")
    if interval is None:
        return None

    portfolio = Portfolio(
        name=name.strip(),
        initial_cash=initial_cash,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        tickers=tickers,
        weights=weights,
        strategy_map=strategy_map,
        start_date=start_date,
        end_date=end_date,
        interval=interval.strip()
    )

    console.print("\n[#26A69A]Portfolio created successfully.[/#26A69A]")
    console.print(f"[#9E9E9E]{portfolio}[/#9E9E9E]")

    return portfolio

def get_allocated_cash(portfolio, ticker):
    index = portfolio.tickers.index(ticker)
    weight = portfolio.weights[index]
    return portfolio.initial_cash * weight