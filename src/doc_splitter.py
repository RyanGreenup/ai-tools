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


def chunk_doc(doc: str, n: int, r: float, verbose: True) -> list[str]:
    """
    Challenges to solve:
        - How to handle code blocks
          - When splitting to ensure the chunk length is less than n
            code blocks should not be separated if this can be avoided
        - Hanging headings
          - If a heading has only subheadings as children, it should be included

    Approach:
        Probably split as per this approach and then merge chunks if possible
        to get the target length.

        If the chunk is too long, first split out the code blocks and then
        split the remaining text.

        The problem with this, loops will be slow in python, I could use
        a separate rust binary, but I think for now it's simpler to use:

            from langchain_community.document_loaders import DirectoryLoader
    """

    chunks: list[str] = []
    lines: list[str] = []
    inside_code_block = False
    for line in doc.splitlines():
        # Test if loop has entered or exited a code block
        if line.startswith("```"):
            inside_code_block = not inside_code_block
        # Check if the line is a heading, if so reset chunk
        if re.match("^#+", line) and not inside_code_block:
            if len(lines) > 0:
                chunks.append("\n".join(lines))
                lines = []
        # Otherwise, add the line to the current chunk
        lines.append(line)
    for i, c in enumerate(chunks):
        c = f"\n{c}".replace("\n", "\n\t")
        print(f"Chunk {i+1}:\t{c}\n")

    # TODO Now go through the chunks and ensure they are less than n long with an overlap
    # of r

    return chunks


doc, target_chunks = make_test_chunk_document()
chunk_document(doc)
