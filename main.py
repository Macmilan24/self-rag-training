import warnings

warnings.filterwarnings("ignore")

import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from src.graph.workflow import app  # The compiled graph
from src.config import Config

console = Console()


def print_header():
    console.clear()
    console.print(
        Panel.fit(
            "[bold cyan]AURA - RAG[/bold cyan]\n[dim]Adaptive User-Reflective Agent[/dim]\n[yellow]Commercial Code of Ethiopia[/yellow]",
            border_style="cyan",
        )
    )


def main():
    print_header()

    # 1. Input Loop
    while True:
        console.print("\n[bold green]Question:[/bold green] (type 'exit' to quit)")
        user_input = Prompt.ask(">")

        if user_input.lower() in ["exit", "quit"]:
            console.print("[yellow]Goodbye![/yellow]")
            break

        # 2. Run the Graph
        inputs = {"question": user_input, "retry_count": 0}

        console.print("\n")
        with console.status(
            "[bold cyan]Thinking & Reflecting...[/bold cyan]", spinner="earth"
        ) as status:

            # We iterate through the stream to update the UI based on what node is running
            final_generation = ""

            try:
                for output in app.stream(inputs):
                    for key, value in output.items():
                        # UI Updates based on Node
                        if key == "retrieve":
                            status.update(
                                f"[bold blue]🔍 Retrieving legal documents...[/bold blue]"
                            )
                        elif key == "grade_documents":
                            status.update(
                                f"[bold magenta]🧐 Critiquing document relevance...[/bold magenta]"
                            )
                        elif key == "transform_query":
                            status.update(
                                f"[bold yellow]🧠 Refining search query...[/bold yellow]"
                            )
                        elif key == "generate":
                            status.update(
                                f"[bold green]✍️ Drafting response...[/bold green]"
                            )
                            final_generation = value.get("generation", "")

            except Exception as e:
                console.print(f"[bold red]Error during execution:[/bold red] {e}")
                continue

        # 3. Final Output
        console.print(
            Panel(Markdown(final_generation), title="Aura Answer", border_style="green")
        )
        console.print(f"[dim]See 'data/logs' for detailed reasoning trace.[/dim]")


if __name__ == "__main__":
    main()
