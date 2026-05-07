from pathlib import Path
import re

PANDA_XML = Path("assets/models/franka_emika_panda/panda.xml")

def main():
    text = PANDA_XML.read_text(encoding="utf-8")

    # Prefix assets/ for any <mesh ... file="..."> that doesn't already start with assets/
    def repl(m):
        before = m.group(1)
        fname = m.group(2)
        after = m.group(3)
        if fname.startswith("assets/") or fname.startswith("assets\\"):
            return m.group(0)
        # Only patch simple filenames (no directories)
        if "/" in fname or "\\" in fname:
            return m.group(0)
        return f'{before}assets/{fname}{after}'

    new_text = re.sub(r'(<mesh\b[^>]*\bfile=")([^"]+)(")', repl, text)

    if new_text == text:
        print("No changes made (maybe already patched).")
    else:
        PANDA_XML.write_text(new_text, encoding="utf-8")
        print("Patched:", PANDA_XML)

if __name__ == "__main__":
    main()
