import json
from typing import Optional

import requests
from tomo2seg.logger import logger

# should contain the webhook_url
SLACK_JSON = "/home/users/jcasagrande/projects/tomo2seg/data/slack.json"


class SlackJsonError(Exception):
    pass


def default_url():

    logger.debug(f"Getting default url from file {SLACK_JSON}")

    try:
        with open(SLACK_JSON, "r") as f:
            slack_json = json.load(f)

    except FileNotFoundError:
        logger.exception(
            f"Please create a json at {SLACK_JSON} with the key `webhook_url`."
        )
        raise SlackJsonError("FileNotFound")

    try:
        return slack_json["webhook_url"]

    except KeyError:
        logger.exception(f"Please create the key `webhook_url` in the slack.json file.")
        raise SlackJsonError("MissingWebhookUrl")


def notify(msg: str, url: Optional[str] = None):
    logger.debug("Sending slack notification.")

    if url is not None:
        logger.debug("A non-default url was given.")

    else:
        try:
            url = default_url()
        except SlackJsonError:
            logger.exception(
                "A notification could not be sent because the webhook url could not be found. "
                "Please correct your slack.json file."
            )

    try:
        resp = requests.post(url=url, json={"text": msg}, timeout=10)
        logger.debug(f"{resp.text=}")

    except Exception as ex:
        logger.exception("Something went wrong in the slack module.")


def notify_finished():
    logger.info("Sending notification of finished training.")
    notify("Training finished!")


def notify_error():
    logger.error("Sending notification error during the training!")
    notify("A problem occurred during the training.")


def notify_exception(exception: Exception, hostname: str = None):
    hostname = "unknown host" if hostname is None else hostname
    logger.exception(f"{exception.__class__.__name__} occurred. ")
    notify(f"{exception.__class__.__name__}: {str(exception)}. {hostname=}")


if __name__ == "__main__":
    notify("test message :*")
