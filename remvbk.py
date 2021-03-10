import os
import sys
from rembg.bg import remove
sys.stdout.buffer.write(remove(sys.stdin.buffer.read()))