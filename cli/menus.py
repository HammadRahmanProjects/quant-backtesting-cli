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
                value=portfolio_id
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