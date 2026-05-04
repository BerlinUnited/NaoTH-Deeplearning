import os
import argparse
import sys
from label_studio_sdk import LabelStudio


def main():
    parser = argparse.ArgumentParser(
        description="Test script to push mock predictions to a Label Studio task."
    )
    parser.add_argument(
        "-t",
        "--task",
        type=int,
        required=True,
        help="The ID of the specific Label Studio Task",
    )
    args = parser.parse_args()

    api_key = os.environ.get("LABELSTUDIO_API_KEY")
    if not api_key:
        print("Error: LABELSTUDIO_API_KEY environment variable is missing.")
        sys.exit(1)

    client = LabelStudio(
        base_url="https://labelstudio-api.berlin-united.com",
        api_key=api_key,
    )

    mock_predictions = [
        {
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "score": 0.95,
            "value": {
                "x": 10.0,
                "y": 10.0,
                "width": 20.0,
                "height": 20.0,
                "rotation": 0,
                "rectanglelabels": ["Ball"],
            },
        },
        {
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "score": 0.88,
            "value": {
                "x": 50.0,
                "y": 30.0,
                "width": 15.0,
                "height": 40.0,
                "rotation": 0,
                "rectanglelabels": ["Robot"],
            },
        },
        {
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "score": 0.75,
            "value": {
                "x": 75.0,
                "y": 75.0,
                "width": 10.0,
                "height": 10.0,
                "rotation": 0,
                "rectanglelabels": ["Nao"],
            },
        },
    ]

    try:
        task = client.tasks.get(id=args.task)
        if hasattr(task, "predictions") and task.predictions:
            for pred in task.predictions:
                print(f"Deleting old prediction {pred.id}...")
                client.predictions.delete(id=pred.id)
    except Exception as e:
        print(f"Note: Could not fetch/delete old predictions: {e}")

    mean_score = sum(p["score"] for p in mock_predictions) / len(mock_predictions)
    try:
        client.predictions.create(
            task=args.task, result=mock_predictions, score=mean_score
        )
        print("done")
    except Exception as e:
        print(f"error: {e}")


if __name__ == "__main__":
    main()
