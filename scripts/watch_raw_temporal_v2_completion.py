"""Keep Windows awake during the research queue and sound once after delivery."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time
import winsound

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/raw_temporal_v2'
STATE = OUT/'completion_notification.json'
CONTINUOUS = 0x80000000
SYSTEM_REQUIRED = 0x00000001


def read(path):
    try: return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError): return {}


def ready():
    status = read(OUT/'validation.json')
    delivery = read(OUT/'delivery_validation.json')
    notebook = ROOT/'notebooks/07_raw_temporal_research_recovery.ipynb'
    if not (status.get('complete') and delivery.get('complete') and
            delivery.get('notebook_refreshed_after_completion')):
        return False
    return (notebook.exists() and hashlib.sha256(notebook.read_bytes()).hexdigest() ==
            delivery.get('notebook', {}).get('sha256'))


def save(status, **extra):
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(status=status, pid=os.getpid(),
        updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **extra), indent=2), encoding='utf-8')
    temporary.replace(STATE)


def main():
    # A named OS mutex prevents duplicate watchers and duplicate sounds.
    kernel = ctypes.windll.kernel32
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    handle = kernel.CreateMutexW(None, False, 'Local\\DiaGraMoRawTemporalV2Completion')
    if not handle: raise ctypes.WinError()
    duplicate = kernel.GetLastError() == 183
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    if duplicate:
        kernel.CloseHandle(handle)
        return
    try:
        if read(STATE).get('status') == 'notified': return
        if not kernel.SetThreadExecutionState(CONTINUOUS | SYSTEM_REQUIRED):
            raise ctypes.WinError()
        save('watching', automatic_system_sleep_prevented=True,
             condition='All phases validated, delivery complete, refreshed notebook hash matched')
        print('WATCHING: automatic system sleep prevented; screen may turn off.', flush=True)
        while not ready(): time.sleep(10)
        # Reserve the single notification before playing it, so restarting this
        # watcher after completion cannot sound again.
        save('notifying', automatic_system_sleep_prevented=False)
        winsound.Beep(1000, 1000)
        save('notified', automatic_system_sleep_prevented=False, sound='1000 Hz, one second')
        print('COMPLETION SOUND PLAYED', flush=True)
    except Exception as error:
        save('error', error=repr(error), automatic_system_sleep_prevented=False)
        raise
    finally:
        kernel.SetThreadExecutionState(CONTINUOUS)
        kernel.CloseHandle(handle)


if __name__ == '__main__': main()
