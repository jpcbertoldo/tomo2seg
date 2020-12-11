from IPython.core.magic import register_cell_magic
import socket

from tomo2seg.slack import notify_exception


@register_cell_magic('slack_exception_notification')
def slack_exception_notification(line, cell):

    try:
        exec(cell)

    except Exception as e:
        hostname = socket.gethostname()
        notify_exception(e, hostname)

        raise e
