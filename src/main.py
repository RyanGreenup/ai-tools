import re
from textwrap import dedent


def make_test_chunk_document() -> tuple[str, list[str]]:
    lorem = dedent("""\
    Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed
    do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
    nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor
    in reprehenderit in voluptate velit esse cillum dolore eu fugiat
    nulla pariatur. Excepteur sint occaecat cupidatat non proident,
    sunt in culpa qui officia deserunt mollit anim id est laborum.
    """)

    one = f"""\
# Heading 1
{lorem}
## Heading 1.1
{lorem}
### Heading 1.1.1
{lorem}
"""

    code_block = f"""\
```markdown
# heading inside code block
{lorem}
## Subheading inside code block
{lorem}
``` """

    two = f"""\
# Heading 2
## Subheading 2.1
{lorem}
## Subheading 2.2
{code_block}
## Subheading 2.3
{lorem}
"""

    three = f"""\
# Heading 3
{lorem}"""

    indented_code = "# indented code\n    # Code\n    # code\n"
    chunks = [dedent(c) for c in [one, two, three]]
    chunks.append(indented_code)
    doc = dedent("\n".join(chunks)).strip()

    return doc, chunks


def test_chunk_document():
    doc, chunks = make_test_chunk_document()
    assert chunk_document(doc) == chunks




def chunk_document(doc: str) -> list[str]:
    # Regular expression pattern for matching markdown headings, using recursive subgroups to ignore anything between tripple backticks ```
    pattern = r"(#.*?(?=((```.*?```|[^`]*?)*?(\n\n#|$))))"

    pattern = re.compile(pattern, re.DOTALL | re.MULTILINE)

    # Find all matches of the pattern and map them to a list
    chunks = pattern.findall(doc)

    # Because of the lookahead, an extra element is included in each tuple, which
    # we don't need. So, we select only the first element from each tuple and
    # remove leading/trailing white space.
    return [chunk[0].strip() for chunk in chunks]



doc, chunks = make_test_chunk_document()
print(doc)

for i, c in enumerate(chunk_document(doc)):
    print(f"Chunk {i+1}:\n{c}\n")





doc, target_chunks = make_test_chunk_document()
chunks: list[str] = []
lines: list[str] = []
inside_code_block = False
for line in doc.splitlines():
    # Test if loop has entered or exited a code block
    if line.startswith("```"):
        inside_code_block = not inside_code_block
    # Check if the line is a heading, if so reset chunk
    if line.startswith("#") and not inside_code_block:
    # if re.match("^#+", line) and not inside_code_block:
        if len(lines) > 0:
            chunks.append("\n".join(lines))
            lines = []
    # Otherwise, add the line to the current chunk
    lines.append(line)
for i, c in enumerate(chunks):
    c = f"\n{c}".replace("\n", "\n\t")
    print(f"Chunk {i+1}:\t{c}\n")

