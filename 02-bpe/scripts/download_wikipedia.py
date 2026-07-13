from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PAGES = {
    "en": "India",
    "hi": "भारत",
    "te": "భారతదేశం",
    "kn": "ಭಾರತ",
}


def fetch(code: str, title: str) -> dict:
    query = urlencode(
        {
            "action": "query",
            "prop": "extracts|revisions",
            "explaintext": "1",
            "redirects": "1",
            "rvprop": "ids|timestamp",
            "format": "json",
            "formatversion": "2",
            "titles": title,
        }
    )
    request = Request(
        f"https://{code}.wikipedia.org/w/api.php?{query}",
        headers={"User-Agent": "IndiaBPEAssignment/1.0"},
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the four Wikipedia corpora.")
    parser.add_argument(
        "--base-dir", type=Path, default=Path(__file__).resolve().parent.parent / "assets"
    )
    parser.add_argument(
        "--responses",
        type=Path,
        help="Read cached india-wikipedia-<language>.json responses from this directory.",
    )
    args = parser.parse_args()

    retrieved_at = datetime.now(timezone.utc).isoformat()
    records = []
    for code, requested_title in PAGES.items():
        if args.responses:
            response_path = args.responses / f"india-wikipedia-{code}.json"
            data = json.loads(response_path.read_text(encoding="utf-8"))
        else:
            data = fetch(code, requested_title)

        page = data["query"]["pages"][0]
        revision = page["revisions"][0]
        file_name = f"wikiepedia-{code}.txt"
        (args.base_dir / file_name).write_text(
            page["extract"].rstrip() + "\n", encoding="utf-8"
        )
        records.append(
            {
                "code": code,
                "file": file_name,
                "requested_title": requested_title,
                "resolved_title": page["title"],
                "page_id": page["pageid"],
                "revision_id": revision["revid"],
                "revision_timestamp": revision["timestamp"],
                "retrieved_at": retrieved_at,
                "source_url": f"https://{code}.wikipedia.org/wiki/{page['title']}",
                "acquisition": "MediaWiki API: prop=extracts|revisions, explaintext=1, redirects=1",
                "cleaning": "API plaintext extraction; whitespace split only; case and punctuation retained",
            }
        )

    (args.base_dir / "wikipedia-sources.json").write_text(
        json.dumps({"pages": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
