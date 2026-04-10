from functools import wraps

from rich.console import Console

from cli.inputs import confirm_action, prompt_text
from cli.menus import (
    prompt_saved_portfolio_menu,
    prompt_saved_portfolio_action_menu,
)
from cli.output import (
    print_backtest_results,
    print_risk_analysis,
    print_optimization_results,
    plot_optimization_curves,
    plot_optimization_heatmap,
    plot_backtest_visuals,
    plot_risk_visuals,
)
from data.fetchers import pull_market_data
from database.portfolio_store import (
    save_portfolio,
    list_portfolios,
    load_portfolio,
    delete_portfolio,
    update_portfolio
)
from engine.backtest_runner import run_backtests
from engine.data_pipeline import process_market_data
from engine.optimization import optimize_portfolio
from engine.portfolio_builder import create_portfolio, edit_portfolio
from engine.strategy_runner import generate_signals
from strategies.registry import AVAILABLE_STRATEGIES

console = Console()

def cancellable_action(func):
    @wraps(func)
    def wrapper(state, *args, **kwargs):
        try:
            return func(state, *args, **kwargs)
        except KeyboardInterrupt:
            console.print("\n[#FF9800]Action cancelled. Returning to main menu.[/#FF9800]")
            return

    return wrapper

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

def _prompt_strategy_params_map(portfolio):
    strategy_params_map = {}

    for ticker in portfolio.tickers:
        strategy_info = portfolio.strategy_map[ticker]
        strategy_name = strategy_info["name"]

        strategy_class = AVAILABLE_STRATEGIES.get(strategy_name)
        if strategy_class is None:
            raise ValueError(f"{ticker}: Unknown strategy '{strategy_name}'.")

        if not hasattr(strategy_class, "get_param_names"):
            raise ValueError(f"{ticker}: Strategy '{strategy_name}' does not define get_param_names().")

        if not hasattr(strategy_class, "get_default_params"):
            raise ValueError(f"{ticker}: Strategy '{strategy_name}' does not define get_default_params().")

        param_names = strategy_class.get_param_names()
        default_params = strategy_class.get_default_params()

        console.print(f"\n[#2962FF]{ticker} — {strategy_name} parameters[/#2962FF]")

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

        strategy_params_map[ticker] = strategy_params

    return strategy_params_map

@cancellable_action
def handle_create_portfolio(state):
    portfolio = create_portfolio()

    if portfolio is None:
        console.print("[#FF9800]Action cancelled. Returning to main menu.[/#FF9800]")
        return

    save_portfolio(portfolio)
    state["current_portfolio"] = portfolio
    state["last_backtest_results"] = None

    console.print("[#26A69A]Portfolio created, saved, and stored in session.[/#26A69A]")

@cancellable_action
def handle_view_saved_portfolios(state):
    while True:
        portfolios = list_portfolios()

        if not portfolios:
            console.print("[#FF9800]No saved portfolios found.[/#FF9800]")
            return

        selected_id = prompt_saved_portfolio_menu(portfolios, "Select a saved portfolio:")

        if selected_id in (None, "BACK"):
            return

        selected_action = prompt_saved_portfolio_action_menu()

        if selected_action in (None, "Back"):
            continue

        if selected_action == "Delete Portfolio":
            confirmed = confirm_action("Delete this portfolio?")
            if confirmed:
                delete_portfolio(selected_id)
                console.print("[red]Portfolio deleted.[/red]")
            continue

        if selected_action == "Load Into Session":
            portfolio = load_portfolio(selected_id)
            if portfolio is None:
                console.print("[red]Failed to load portfolio.[/red]")
                continue
            state["current_portfolio"] = portfolio
            state["last_backtest_results"] = None
            console.print("[#26A69A]Portfolio loaded into session.[/#26A69A]")
            continue

        if selected_action == "Edit Portfolio":
            portfolio = load_portfolio(selected_id)
            if portfolio is None:
                console.print("[red]Failed to load portfolio.[/red]")
                continue
            handle_edit_portfolio(state, selected_id, portfolio)
            continue

@cancellable_action
def handle_view_current_portfolio(state):
    portfolio = state.get("current_portfolio")

    if portfolio is None:
        console.print("[#FF9800]No portfolio loaded in the current session.[/#FF9800]")
        return

    console.print("\n[#2962FF]Current Portfolio[/#2962FF]")
    console.print(portfolio)

@cancellable_action
def handle_run_backtest(state):
    portfolio = state.get("current_portfolio")

    if portfolio is None:
        console.print("[red]No portfolio loaded in the current session.[/red]")
        return

    strategy_params_map = _prompt_strategy_params_map(portfolio)
    if strategy_params_map is None:
        console.print("[#FF9800]Action cancelled. Returning to main menu.[/#FF9800]")
        return

    console.print("\n[#FF9800]Running backtest...[/#FF9800]")

    market_data = pull_market_data(portfolio)
    processed_data = process_market_data(market_data)
    signals_data = generate_signals(processed_data, portfolio, strategy_params_map)  
    backtest_results = run_backtests(signals_data, portfolio)

    state["last_backtest_results"] = backtest_results

    console.print("[#26A69A]Backtest complete.[/#26A69A]")
    print_backtest_results(backtest_results, portfolio)
    plot_backtest_visuals(backtest_results, portfolio)

@cancellable_action
def handle_view_results(state):
    portfolio = state.get("current_portfolio")
    backtest_results = state.get("last_backtest_results")

    if portfolio is None:
        console.print("[red]No portfolio available in the current session.[/red]")
        return

    if backtest_results is None:
        console.print("[#FF9800]No backtest results available yet.[/#FF9800]")
        return

    print_backtest_results(backtest_results, portfolio)

@cancellable_action
def handle_run_risk_analysis(state):
    portfolio = state.get("current_portfolio")

    if portfolio is None:
        console.print("[red]No portfolio loaded in the current session.[/red]")
        return

    strategy_params_map = _prompt_strategy_params_map(portfolio)
    if strategy_params_map is None:
        console.print("[#FF9800]Action cancelled. Returning to main menu.[/#FF9800]")
        return

    console.print("\n[#FF9800]Running risk analysis...[/#FF9800]")

    market_data = pull_market_data(portfolio)
    processed_data = process_market_data(market_data)
    signals_data = generate_signals(processed_data, portfolio, strategy_params_map)
    backtest_results = run_backtests(signals_data, portfolio)

    state["last_backtest_results"] = backtest_results

    console.print("[#26A69A]Risk analysis complete.[/#26A69A]")
    print_risk_analysis(backtest_results, portfolio)
    plot_risk_visuals(backtest_results, portfolio)

@cancellable_action
def handle_run_optimization(state):
    portfolio = state.get("current_portfolio")

    if portfolio is None:
        console.print("[red]No portfolio loaded in the current session.[/red]")
        return

    console.print("\n[#FF9800]Running optimization...[/#FF9800]")

    market_data = pull_market_data(portfolio)
    processed_data = process_market_data(market_data)
    optimization_results = optimize_portfolio(processed_data, portfolio)

    console.print("[#26A69A]Optimization complete.[/#26A69A]")
    print_optimization_results(optimization_results)
    plot_optimization_curves(optimization_results, top_n=10)
    plot_optimization_heatmap(optimization_results, metric="sharpe_ratio")

@cancellable_action
def handle_edit_portfolio(state, portfolio_id, portfolio):
    updated_portfolio = edit_portfolio(portfolio)

    if updated_portfolio is None:
        console.print("[#FF9800]Edit cancelled.[/#FF9800]")
        return

    update_portfolio(portfolio_id, updated_portfolio)
    console.print("[#26A69A]Portfolio updated successfully.[/#26A69A]")

    # If the edited portfolio was the active session one, update it
    current = state.get("current_portfolio")
    if current is not None and current.name == portfolio.name:
        state["current_portfolio"] = updated_portfolio
        state["last_backtest_results"] = None

def handle_exit(state):
    confirmed = confirm_action("Are you sure you want to exit?")

    if not confirmed:
        return

    console.print("[#EF5350]Exiting...[/#EF5350]")
    state["running"] = False

MENU_ACTIONS = {
    "Create Portfolio": handle_create_portfolio,
    "View Saved Portfolios": handle_view_saved_portfolios,
    "View Current Portfolio": handle_view_current_portfolio,
    "Run Backtest": handle_run_backtest,
    "View Results": handle_view_results,
    "Run Risk Analysis": handle_run_risk_analysis,
    "Run Optimization": handle_run_optimization,
    "Exit": handle_exit,
}