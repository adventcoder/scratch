
import json
import time
import sys
import struct
import argparse
import signal
from ctypes import windll, create_string_buffer
from collections import namedtuple

Message = namedtuple('Message', ['time', 'author', 'text'])

def format_time(time):
    s = time % 60
    m = int(time) // 60 % 60
    ts = '%02d:%02d' % (m, s)
    h = int(time) // 3600
    if h != 0:
        ts = str(h) + ':' + ts
    return ts

def parse_time(ts):
    parts = ts.split(':', 3)
    time = float(parts[-1])
    if len(parts) > 1:
        time += 60 * int(parts[-2])
        if len(parts) > 2:
            time += 3600 * int(parts[-3])
    return time

def play(args):
    messages = load_messages(args.filename)
    print("Playing", args.filename)
    start_time = time.monotonic()
    for message in messages:
        if message.time < args.offset:
            continue
        current_time = time.monotonic() - start_time + args.offset
        if message.time > current_time:
            time.sleep(message.time - current_time)
        print_message(message, args)

def print_message(message, args):
    console = get_console(args)
    console.write(format_time(message.time), color = 8)
    console.write(' ')
    console.write(sanitize(message.author), color = 6)
    console.write(': ')
    console.write(sanitize(message.text))
    console.write('\n')

def sanitize(text):
    return text.encode('ascii', 'replace').decode()

def get_console(args):
    if sys.stdout.isatty() and not args.raw:
        if sys.platform == 'win32':
            return Win32Console()
        else:
            return AnsiConsole()
    return RawConsole()

class RawConsole:
    def write(self, text, color = None):
        sys.stdout.write(text)

class AnsiConsole:
    def write(self, text, color = None):
        if color is None:
            sys.stdout.write(text)
        else:
            sys.stdout.write(self.escape_code((30 if color & 0x8 else 90) + (color & 0x7)))
            sys.stdout.write(text)
            sys.stdout.write(self.escape_code(0))

    def escape_code(self, code):
        return '\033[%dm' % (code)

class Win32Console:
    def __init__(self):
        self.hout = windll.kernel32.GetStdHandle((-11) & 0xFFFFFFFF)
        csbi = create_string_buffer(22)
        windll.kernel32.GetConsoleScreenBufferInfo(self.hout, csbi)
        _, _, _, _, self.default_attrs, _, _, _, _, _, _ = struct.unpack('hhhhHhhhhhh', csbi.raw)

    def write(self, text, color = None):
        if color is None:
            sys.stdout.write(text)
        else:
            sys.stdout.flush()
            windll.kernel32.SetConsoleTextAttribute(self.hout, self.default_attrs & (~0xF) | color)
            sys.stdout.write(text)
            sys.stdout.flush()
            windll.kernel32.SetConsoleTextAttribute(self.hout, self.default_attrs)

def load_messages(filename):
    messages = []
    with open(filename, 'r') as file:
        for action in json.load(file):
            if action['action_type'] == 'add_chat_item':
                time = action['time_in_seconds']
                author = action['author']['name']
                text = action['message']
                messages.append(Message(time, author, text))
    messages.sort(key = lambda m: m.time)
    return messages

def signal_handler(signal, frame):
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar = 'FILENAME', help = 'the json chat file')
    parser.add_argument('--offset', metavar = 'OFFSET', type = parse_time, default = 0, help = 'offset to start of chat (H:M:S)')
    parser.add_argument('--raw', action = 'store_true', help = 'force raw output')
    play(parser.parse_args())

if __name__ == '__main__':
    main()
