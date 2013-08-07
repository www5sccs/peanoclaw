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

def main():
  try:
    from subprocess import call
    with time_limit(args.timeout):
      return call('python ' + args.script, shell=True)
  except TimeoutException, msg:
      print "Timed out!"
      
if __name__=="__main__":
  exit(main())