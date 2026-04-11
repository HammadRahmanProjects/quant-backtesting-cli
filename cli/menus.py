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
    """
    Prompt the user to select a position sizing method.
    Returns the string value of the selected SizingMethod.
    """
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
    """Returns (risk_pct, stop_loss_pct) or None if cancelled."""
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
            "risk_pct":     float(risk_pct_raw),
            "stop_loss_pct": float(stop_pct_raw),
        }
    except ValueError:
        return {"risk_pct": 0.02, "stop_loss_pct": 0.05}


def prompt_volatility_target_params():
    """Returns sizing params dict or None if cancelled."""
    target_raw = questionary.text(
        "Annualized volatility target (e.g. 0.15 for 15%):"
    ).ask()
    if target_raw is None:
        return None

    try:
        return {"target_vol_pct": float(target_raw)}
    except ValueError:
        return {"target_vol_pct": 0.15}