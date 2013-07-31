import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

from argparse import ArgumentParser
parser = ArgumentParser(description='Starts script with a timeout.')
parser.add_argument('timeout', type=int, help='Timeout in seconds')
parser.add_argument('script', help='Script to be started.')
args = parser.parse_args()

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException, "Timed out!"
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

try:
  import os
  with time_limit(args.timeout):
      os.system('python ' + args.script)
except TimeoutException, msg:
    print "Timed out!"