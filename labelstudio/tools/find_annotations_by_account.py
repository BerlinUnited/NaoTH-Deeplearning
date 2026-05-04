# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "python-dotenv",
#     "requests",
# ]
# ///

import argparse
import json
import os
import sys

import requests
from dotenv import load_dotenv


def get_ls_session(base_url: str, api_key: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"Authorization": f"Token {api_key}"})
    session.base_url = base_url
    return session


def get_all_projects(session: requests.Session) -> list[dict]:
    url = f"{session.base_url}/api/projects"
    projects = []
    page = 1
    while True:
        resp = session.get(url, params={"page": page, "page_size": 100})
        resp.raise_for_status()
        data = resp.json()
        projects.extend(data["results"])
        if not data["next"]:
            break
        page += 1
    return projects


def get_user_id(session: requests.Session, username: str) -> int | None:
    """Lookup user ID by username or email via the /api/users endpoint."""
    resp = session.get(f"{session.base_url}/api/users", params={"search": username})
    resp.raise_for_status()
    for user in resp.json():
        if user.get("username") == username or user.get("email") == username:
            return user["id"]
    return None


def get_annotations_for_project(
    session: requests.Session, project_id: int, user_id: int
) -> list[dict]:
    """Fetch all tasks for a project and collect annotations created by user_id."""
    url = f"{session.base_url}/api/projects/{project_id}/tasks"
    annotations = []
    page = 1
    while True:
        resp = session.get(url, params={"page": page, "page_size": 100})
        if resp.status_code == 404:
            break
        resp.raise_for_status()
        tasks = resp.json()
        if not tasks:
            break
        for task in tasks:
            for ann in task.get("annotations", []):
                if ann.get("completed_by") == user_id:
                    annotations.append(ann)
        page += 1
    return annotations


def load_env_file(env_file: str | None) -> None:
    """Load environment variables from a .sh or .env file into os.environ."""
    if not env_file:
        return
    env_path = os.path.expanduser(env_file)
    if not os.path.isfile(env_path):
        print(f"WARNING: Env file '{env_path}' not found.", file=sys.stderr)
        return
    if env_path.endswith(".sh"):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export ") :]
                if "=" in line and not line.startswith("#"):
                    key, _, value = line.partition("=")
                    value = value.strip("'\"")
                    os.environ[key] = value
    else:
        load_dotenv(env_path, override=True)


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env-file", default=os.getenv("ENV_FILE"))
    pre_args, _ = pre_parser.parse_known_args()
    load_env_file(pre_args.env_file)

    parser = argparse.ArgumentParser(
        description="Find all annotations for a Label Studio user."
    )
    parser.add_argument(
        "username",
        help="The Label Studio username",
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        type=int,
        default=None,
        help="Optional list of project IDs. If not provided, all projects are searched.",
    )
    parser.add_argument(
        "--url",
        default=os.getenv("LABELSTUDIO_API_URL"),
        help="Label Studio API URL (or set LABELSTUDIO_API_URL)",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("LABELSTUDIO_API_KEY"),
        help="Label Studio API token (or set LABELSTUDIO_API_KEY)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output as JSON instead of tabular",
    )
    parser.add_argument(
        "--env-file",
        default=os.getenv("ENV_FILE"),
        help="Path to a .sh/.env file to load environment variables",
    )

    args = parser.parse_args()

    if not args.url:
        print("ERROR: --url or LABELSTUDIO_API_URL is required.", file=sys.stderr)
        sys.exit(1)
    if not args.token:
        print("ERROR: --token or LABELSTUDIO_API_KEY is required.", file=sys.stderr)
        sys.exit(1)

    session = get_ls_session(args.url, args.token)

    user_id = get_user_id(session, args.username)
    if user_id is None:
        print(f"ERROR: User '{args.username}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"User '{args.username}' (ID: {user_id}) found.\n")

    if args.projects:
        projects = [{"id": pid, "title": f"Project {pid}"} for pid in args.projects]
    else:
        projects = get_all_projects(session)
        print(f"{len(projects)} projects found.\n")

    all_results: list[dict] = []

    for project in projects:
        pid = project["id"]
        ptitle = project.get("title", f"Project {pid}")
        annotations = get_annotations_for_project(session, pid, user_id)
        for ann in annotations:
            all_results.append(
                {
                    "project_id": pid,
                    "project_title": ptitle,
                    "annotation_id": ann["id"],
                    "task_id": ann["task"],
                    "updated_at": ann.get("updated_at", ""),
                    "lead": ann.get("lead", None),
                }
            )
        if annotations:
            print(f"  {ptitle} ({pid}): {len(annotations)} annotation(s)")

    print(f"\nTotal: {len(all_results)} annotation(s) found.")

    if args.as_json:
        print(json.dumps(all_results, indent=2, ensure_ascii=False))


def cli() -> None:
    main()


if __name__ == "__main__":
    main()
