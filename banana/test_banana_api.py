import argparse
import os

import banana_dev as banana
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("BANANA_API_KEY")
model_key = os.getenv("BANANA_MODEL_KEY")


def generate_predictions(args):
    print(banana.run(api_key, model_key, {"prompt": args.prompt}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default="The service is terrible, the staff seem to be generally clueless, the management is inclined to blame the staff for their own mistakes, and there's no sense of FAST in their fast food.",
        type=str,
        help="Prompt",
    )
    args = parser.parse_args()
    generate_predictions(args=args)
