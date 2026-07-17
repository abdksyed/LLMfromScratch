from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from markdownify import markdownify


PAGES = {
    "en": {"name": "English", "title": "India"},
    "hi": {"name": "Hindi", "title": "भारत"},
    "te": {"name": "Telugu", "title": "భారతదేశం"},
    "kn": {"name": "Kannada", "title": "ಭಾರತ"},
}
USER_AGENT = "IndiaBPEAssignment/2.0 (faithful tokenizer corpus)"


def faithful_units(text: str) -> int:
    """Match the assignment's Unicode letter/mark/number unit policy."""
    total = 0
    in_word = False
    for character in text:
        is_word = unicodedata.category(character)[0] in {"L", "M", "N"}
        if is_word:
            if not in_word:
                total += 1
            in_word = True
        else:
            in_word = False
            if not character.isspace():
                total += 1
    return total


def fetch_html(code: str, title: str) -> tuple[str, str, dict[str, str]]:
    url = f"https://{code}.wikipedia.org/api/rest_v1/page/html/{quote(title)}"
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=45) as response:
        html = response.read().decode("utf-8")
        headers = {
            key: value
            for key in ("etag", "last-modified", "content-revision-id")
            if (value := response.headers.get(key))
        }
    return html, url, headers


def make_links_absolute(node: BeautifulSoup, page_url: str) -> None:
    for tag in node.find_all(["a", "img", "source"]):
        attribute = "href" if tag.name == "a" else "src"
        value = tag.get(attribute)
        if value:
            tag[attribute] = urljoin(page_url, value)


def remove_non_content_nodes(node: BeautifulSoup, soup: BeautifulSoup) -> None:
    for tag in node.find_all(["script", "style", "meta"]):
        tag.decompose()

    for tag in node.find_all("link"):
        relations = " ".join(tag.get("rel") or ())
        href = tag.get("href")
        if "mw:PageProp/Category" in relations and href:
            tag.replace_with(soup.new_string(f"\nCategory: {href}\n"))
        else:
            tag.decompose()


def normalize_markdown(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


def html_to_markdown(html: str, page_url: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    body = soup.find("body") or soup
    remove_non_content_nodes(body, soup)
    make_links_absolute(body, page_url)
    return normalize_markdown(
        markdownify(
            str(body),
            heading_style="ATX",
            bullets="-",
            strip=["span"],
        )
    )


def build_language(code: str, output_dir: Path) -> dict[str, object]:
    page = PAGES[code]
    html, source_url, response_headers = fetch_html(code, page["title"])
    markdown = html_to_markdown(html, source_url)
    output_path = output_dir / f"wikipedia-{code}.txt"
    output_path.write_text(markdown, encoding="utf-8")

    encoded = markdown.encode("utf-8")
    return {
        "code": code,
        "name": page["name"],
        "title": page["title"],
        "file": output_path.name,
        "source_url": source_url,
        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "acquisition": "Wikipedia REST HTML converted to Markdown",
        "cleaning": "Removed technical script/style/meta/link nodes; preserved visible links, URLs, tables, references, media links, navboxes, categories, punctuation, and symbols",
        "response_headers": response_headers,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "characters": len(markdown),
        "words": len(re.findall(r"\S+", markdown)),
        "faithful_units": faithful_units(markdown),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build wiki-faithful Markdown snapshots for the India tokenizer."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "assets",
    )
    args = parser.parse_args()

    output_dir = args.base_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [build_language(code, output_dir) for code in PAGES]
    metadata = {
        "variant": "wiki_faithful_markdown",
        "unit_policy": "One contiguous Unicode letter/mark/number run, or one visible non-whitespace punctuation/symbol character",
        "pages": records,
    }
    (args.base_dir / "wikipedia-sources.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for record in records:
        print(
            f"{record['code']}: {record['faithful_units']} faithful units / "
            f"{record['bytes']} bytes"
        )


if __name__ == "__main__":
    main()
