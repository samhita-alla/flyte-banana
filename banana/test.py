import argparse

import requests


def generate_predictions(args):
    model_inputs = {"prompt": args.prompt}
    res = requests.post(args.url, json=model_inputs)
    print(res.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", default="http://localhost:8000/", type=str, help="API endpoint URL"
    )
    parser.add_argument(
        "--prompt",
        default="The service is terrible, the staff seem to be generally clueless, the management is inclined to blame the staff for their own mistakes, and there's no sense of FAST in their fast food.",
        type=str,
        help="Prompt",
    )
    args = parser.parse_args()
    generate_predictions(args=args)
