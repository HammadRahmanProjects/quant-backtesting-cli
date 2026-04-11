from rich.console import Console

from cli.actions import MENU_ACTIONS
from cli.inputs import confirm_action
from cli.menus import prompt_main_menu
from database.db import initialize_database
from logger import setup_logging

console = Console()

def main():
    setup_logging()
    initialize_database()

    state = {
        "running": True,
        "current_portfolio": None,
        "last_backtest_results": None,
    }

    console.print("\n[#2962FF]=== Quant Backtester CLI ===[/#2962FF]")

    while state["running"]:
        try:
            choice = prompt_main_menu()

            if choice is None:
                if confirm_action("Exit application?"):
                    console.print("\n[#EF5350]Exiting...[/#EF5350]")
                    break
                continue

            action = MENU_ACTIONS.get(choice)

            if action is None:
                console.print(f"[red]Unknown option: {choice}[/red]")
                continue

            action(state)

        except KeyboardInterrupt:
            if confirm_action("Exit application?"):
                console.print("\n[#EF5350]Exiting...[/#EF5350]")
                break

if __name__ == "__main__":
    main()