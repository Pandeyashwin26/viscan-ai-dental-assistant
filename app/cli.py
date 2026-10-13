"""Interactive CLI:  python -m app.cli   (type 'dictate: <text>' for dictation, 'id P001' to resolve ambiguity)"""
import json
from .assistant import Assistant, Session
from .llm import get_llm
from .store import Store

def main():
    a, s = Assistant(Store(), get_llm()), Session()
    print(f"LLM: {a.llm.name}. Ctrl-C to quit.")
    while True:
        line = input("> ").strip()
        if line.lower().startswith("id "):
            out = a.handle(s, "", line[3:].strip())
        elif line.lower().startswith("dictate:"):
            out = a.dictate(s, line[8:].strip())
        else:
            out = a.handle(s, line)
        print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
