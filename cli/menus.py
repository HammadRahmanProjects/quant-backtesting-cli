import questionary

def prompt_main_menu():
    return questionary.select(
        "Select an option:",
        choices=[
            "Create Portfolio",
            "View Saved Portfolios",
            "View Current Portfolio",
            "Run Backtest",
            "View Results",
            "Run Risk Analysis",
            "Run Optimization",
            "Run Walk-Forward Validation",
            "Exit",
        ],
        pointer="➤",
    ).ask()


def prompt_saved_portfolio_menu(portfolios, title):
    if not portfolios:
        return None

    choices = []
    for display_index, (portfolio_id, name, start_date, end_date, interval) in enumerate(portfolios, start=1):
        choices.append(
            questionary.Choice(
                title=f"{display_index}. {name} | {start_date} → {end_date} | {interval}",
                value=portfolio_id,
            )
        )
    choices.append(questionary.Choice(title="Back", value="BACK"))

    return questionary.select(
        title,
        choices=choices,
        pointer="➤",
    ).ask()


def prompt_saved_portfolio_action_menu():
    return questionary.select(
        "Select an action:",
        choices=[
            "Load Into Session",
            "Edit Portfolio",
            "Delete Portfolio",
            "Back",
        ],
        pointer="➤",
    ).ask()


def prompt_sizing_method():
    return questionary.select(
        "Select position sizing method:",
        choices=[
            questionary.Choice(
                title="Full Portfolio  (all-in on every signal)",
                value="full_portfolio",
            ),
            questionary.Choice(
                title="Fixed Fractional  (risk fixed % per trade)",
                value="fixed_fractional",
            ),
            questionary.Choice(
                title="Volatility Target  (scale by inverse volatility)",
                value="volatility_target",
            ),
            questionary.Choice(
                title="Kelly Criterion  (mathematically optimal sizing)",
                value="kelly",
            ),
        ],
        pointer="➤",
    ).ask()


def prompt_fixed_fractional_params():
    risk_pct_raw = questionary.text(
        "Risk per trade as fraction of portfolio (e.g. 0.02 for 2%):"
    ).ask()
    if risk_pct_raw is None:
        return None

    stop_pct_raw = questionary.text(
        "Stop loss as fraction of price (e.g. 0.05 for 5%):"
    ).ask()
    if stop_pct_raw is None:
        return None

    try:
        return {
            "risk_pct"     : float(risk_pct_raw),
            "stop_loss_pct": float(stop_pct_raw),
        }
    except ValueError:
        return {"risk_pct": 0.02, "stop_loss_pct": 0.05}


def prompt_volatility_target_params():
    target_raw = questionary.text(
        "Annualized volatility target (e.g. 0.15 for 15%):"
    ).ask()
    if target_raw is None:
        return None

    try:
        return {"target_vol_pct": float(target_raw)}
    except ValueError:
        return {"target_vol_pct": 0.15}


def prompt_walk_forward_params():
    """
    Prompt user for walk-forward validation parameters.
    Returns dict of params or None if cancelled.
    """
    console_msg = (
        "\n[#9E9E9E]For daily data: train=756 (3yr), test=252 (1yr) "
        "are standard industry defaults.[/#9E9E9E]"
    )

    import questionary as q
    from rich.console import Console
    Console().print(console_msg)

    train_raw = q.text("Training window size (bars):").ask()
    if train_raw is None:
        return None

    test_raw = q.text("Test window size (bars):").ask()
    if test_raw is None:
        return None

    step_raw = q.text(
        "Step size (bars) [leave blank to use test window size]:"
    ).ask()
    if step_raw is None:
        return None

    ranking = q.select(
        "Ranking metric for optimization:",
        choices=[
            "sharpe_ratio",
            "calmar_ratio",
            "sortino_ratio",
            "total_return",
        ],
        pointer="➤",
    ).ask()
    if ranking is None:
        return None

    try:
        train  = int(train_raw.strip())
        test   = int(test_raw.strip())
        step   = int(step_raw.strip()) if step_raw.strip() else test
        return {
            "train_period"  : train,
            "test_period"   : test,
            "step_size"     : step,
            "ranking_metric": ranking,
        }
    except ValueError:
        return {
            "train_period"  : 756,
            "test_period"   : 252,
            "step_size"     : 252,
            "ranking_metric": "sharpe_ratio",
        }