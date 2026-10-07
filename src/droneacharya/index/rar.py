"""RAR access through UnRAR: header listings and single-member reads, nothing unpacked to disk."""
import shutil
import subprocess

UNRAR = shutil.which("UnRAR") or r"C:\Program Files\WinRAR\UnRAR.exe"


def parse_unrar_lt(text):
    """File entries of `UnRAR lt -v` output: name, type, size, crc32, modified (all strings)."""
    entries, current = [], {}
    for line in text.splitlines():
        key, sep, value = line.strip().partition(": ")
        if not sep:
            continue
        if key == "Name":
            current = {"name": value}
            entries.append(current)
        elif current and key in ("Type", "Size", "CRC32", "Modified"):
            current[key.lower()] = value
    return [e for e in entries if e.get("type") == "File"]


def list_rar(path):
    result = subprocess.run([UNRAR, "lt", "-v", str(path)], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", check=True)
    return parse_unrar_lt(result.stdout)


def read_member(path, member):
    """Bytes of one member, streamed through UnRAR (for small members such as XML)."""
    return subprocess.run([UNRAR, "p", "-inul", str(path), member], capture_output=True, check=True).stdout
