#!/usr/bin/env python3
"""Clean a Wikipedia XML dump into plain-text documents, one per article, separated by 0x1E.

Usage: python3 -I wiki_clean.py DUMP.xml.bz2 OUT.bin
Reads the compressed dump as a stream (standard library only). Keeps main-namespace,
non-redirect articles; drops templates, references, tables, files, categories, and markup;
keeps prose paragraphs of at least 40 characters. Deterministic: the same dump gives the same bytes.
Text is CC BY-SA 4.0 (Wikipedia contributors); see docs/data.md for attribution.
"""
import bz2
import html
import re
import sys
import xml.etree.ElementTree as ET

REF = re.compile(r"<ref[^>]*?/>|<ref[^>]*?>.*?</ref>", re.S | re.I)
COMMENT = re.compile(r"<!--.*?-->", re.S)
TAG = re.compile(r"</?[a-zA-Z][^>]*>")
TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
TABLE = re.compile(r"\{\|.*?\|\}", re.S)
FILE = re.compile(r"\[\[(?:File|Image|Category|Media):[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]", re.I)
LINK_PIPE = re.compile(r"\[\[(?:[^\]|]*\|)+([^\]]*)\]\]")
LINK = re.compile(r"\[\[([^\]]*)\]\]")
EXT = re.compile(r"\[(?:https?:)?//\S+\s+([^\]]*)\]|\[(?:https?:)?//\S+\]")
QUOTES = re.compile(r"'{2,}")
HEADING = re.compile(r"^=+\s*(.*?)\s*=+\s*$")


def strip_nested(text):
    for _ in range(12):
        new = TEMPLATE.sub("", text)
        if new == text:
            break
        text = new
    return text


def clean(text):
    text = COMMENT.sub("", text)
    text = REF.sub("", text)
    text = TABLE.sub("", text)
    text = strip_nested(text)
    text = FILE.sub("", text)
    text = LINK_PIPE.sub(r"\1", text)
    text = LINK.sub(r"\1", text)
    text = EXT.sub(lambda m: m.group(1) or "", text)
    text = TAG.sub("", text)
    text = QUOTES.sub("", text)
    text = html.unescape(text).replace("\xa0", " ").replace("\u2009", " ")
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or HEADING.match(line):
            continue
        line = line.lstrip("*#: ").strip()
        if len(line) >= 40 and "{{" not in line and "}}" not in line and "[[" not in line and "|" not in line:
            out.append(line)
    return "\n".join(out)


def main(src, dst):
    n = kept = 0
    with bz2.open(src, "rb") as f, open(dst, "wb") as out:
        title = ns = text = None
        redirect = False
        for _, el in ET.iterparse(f, events=("end",)):
            tag = el.tag.rsplit("}", 1)[-1]
            if tag == "title":
                title = el.text
            elif tag == "ns":
                ns = el.text
            elif tag == "redirect":
                redirect = True
            elif tag == "text":
                text = el.text or ""
            elif tag == "page":
                n += 1
                if ns == "0" and not redirect and text:
                    doc = clean(text)
                    if len(doc) >= 500:
                        out.write(doc.replace("\x1e", " ").encode("utf-8"))
                        out.write(b"\x1e")
                        kept += 1
                title = ns = text = None
                redirect = False
                el.clear()
            elif tag in ("revision", "contributor"):
                el.clear()
    print(f"{kept} articles kept of {n} pages", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
