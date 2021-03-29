"""
Huge kudos to https://stackoverflow.com/a/40135960
"""
import socket

from IPython.core.ultratb import AutoFormattedTB
from tomo2seg import slack

# formatter for making the tracebacks into strings
itb = AutoFormattedTB(mode="Plain", tb_offset=1)


# this function will be called on exceptions in any cell
def custom_exc(shell, etype, evalue, tb, tb_offset=None):

    # still show the error within the notebook, don't just swallow it
    shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)

    # grab the traceback and make it into a list of strings
    stb = itb.structured_traceback(etype, evalue, tb)
    sstb = itb.stb2text(stb)

    hostname = socket.gethostname()
    sstb = f"An exception occurred in {hostname=}\n\n" + sstb

    slack.notify(sstb)
