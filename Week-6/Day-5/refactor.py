import re

with open('d:/netixsol/Week-6/Day-5/afl_assistant_core.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Slice off E2E tests
cutoff = content.find('# ══════════════════════════════════════════════════════════════════════════════\n# 5.  CONVERSATION DEFINITIONS')
if cutoff != -1:
    content = content[:cutoff]

import_idx = content.find('from google import genai')

injection = """
import logging
import json
import concurrent.futures

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName
        }
        if hasattr(record, "extra_data"):
            log_record.update(record.extra_data)
        return json.dumps(log_record)

logger = logging.getLogger("afl_assistant")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(JSONFormatter())
    logger.addHandler(ch)

def run_with_timeout(func, args=(), kwargs={}, timeout_sec=5.0):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            logger.error("Tool execution timed out", extra={"extra_data": {"tool": func.__name__, "timeout_sec": timeout_sec}})
            raise TimeoutError(f"Execution of {func.__name__} timed out after {timeout_sec}s")
"""

new_content = content[:import_idx] + injection + '\n' + content[import_idx:]

with open('d:/netixsol/Week-6/Day-5/afl_assistant_core.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
print('Sliced core file and added logging/timeout utils.')
