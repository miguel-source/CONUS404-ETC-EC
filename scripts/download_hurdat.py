from __future__ import annotations

import html.parser
import ssl
import sys
import urllib.parse
import urllib.request
from pathlib import Path


class LinkParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


def main() -> None:
    index_url = "https://www.nhc.noaa.gov/data/hurdat/"
    final_year = sys.argv[2] if len(sys.argv) > 2 else "2025"
    context = ssl.create_default_context()
    index = urllib.request.urlopen(index_url, context=context).read().decode("utf-8")
    parser = LinkParser()
    parser.feed(index)
    candidates = [
        href
        for href in parser.links
        if href.startswith(f"hurdat2-1851-{final_year}") and href.endswith(".txt")
    ]
    if not candidates:
        raise RuntimeError(f"No {final_year} HURDAT2 text link found")
    href = sorted(candidates)[-1]
    url = urllib.parse.urljoin(index_url, href)
    data = urllib.request.urlopen(url, context=context).read()
    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    print(f"Downloaded {url} to {output} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
