try:
    import argparse
except ImportError:
    input("'argparse' not found. May I install it for you? (yY/nN) ")
    if input().lower() in ["y", "yes"]:
        import subprocess

        subprocess.run(["pip", "install", "argparse"])
        import argparse
    else:
        exit(1)

try:
    from rich.markdown import Markdown
    from rich.console import Console
except ImportError:
    input("'rich' not found. May I install it for you? (yY/nN) ")
    if input().lower() in ["y", "yes"]:
        import subprocess

        subprocess.run(["pip", "install", "rich"])
        from rich.markdown import Markdown
        from rich.console import Console
    else:
        exit(1)

try:
    from pathlib import Path
except ImportError:
    input("'pathlib' not found. May I install it for you? (yY/nN) ")
    if input().lower() in ["y", "yes"]:
        import subprocess

        subprocess.run(["pip", "install", "pathlib"])
        from pathlib import Path
    else:
        exit(1)

console = Console()

filetype_comments = {
    "tex": "%",
    "py": "#",
    "sh": "#",
}

output_markdown_file = "todo.md"
todo_keywords = ["TODO", "Todo", "todo", "ToDo"]


def make_keywords(filetype: str) -> list:
    comment_prefix = filetype_comments[filetype]
    return [f"{comment_prefix} {tag}" for tag in todo_keywords] + [
        f"{comment_prefix}{tag}" for tag in todo_keywords
    ]


def find_todos(
    root_dir: Path, filetype: str = "tex", todo_keywords: list = None
) -> dict:
    todos = {}
    console.print(f"Looking for TODO tasks in {filetype} files 👀")
    for file_path in root_dir.rglob(f"*.{filetype}"):
        with file_path.open("r", encoding="utf-8") as file:
            lines = file.readlines()
            todo_comments = [
                (line.strip(), line_number)
                for line, line_number in zip(lines, range(1, len(lines) + 1))
                if any(keyword in line for keyword in todo_keywords)
            ]
            if todo_comments:
                # Use relative path from the project directory for better readability
                relative_path = file_path.relative_to(root_dir)
                todos[str(relative_path)] = todo_comments
    return todos


def write_markdown(todos, output_file, todo_keywords=todo_keywords):
    with open(output_file, "w", encoding="utf-8") as md_file:
        for file, comments in todos.items():
            md_file.write(f"## {file}\n")
            for comment, line_number in comments:
                # Remove the todo keyword and the leading space if there is any
                for keyword in todo_keywords:
                    comment = comment.replace(keyword, "").strip()
                # Remove leading colon if there is any
                comment = comment.lstrip(":").strip()
                md_file.write(f"- [ ] {comment} - Line {line_number}\n")
            md_file.write("\n")


def argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract TODO tasks from a project to a markdown file"
    )
    parser.add_argument(
        "project_directory",
        type=Path,
        help="The root directory of the project",
    )
    parser.add_argument(
        "--filetype",
        "-f",
        type=str,
        default="tex",
        help="The file type to look for TODO tasks",
    )
    return parser.parse_args()


def main():
    args = argparser()
    project_directory = args.project_directory
    filetype = args.filetype
    keywords = make_keywords(filetype)

    todos = find_todos(project_directory, filetype, keywords)
    write_markdown(todos, output_markdown_file, keywords)
    console.print(f"TODO tasks have been extracted to {output_markdown_file} 📝")

    with open(output_markdown_file, "r", encoding="utf-8") as file:
        markdown = file.read()
        md = Markdown(markdown)
        console.print(md)


if __name__ == "__main__":
    main()
