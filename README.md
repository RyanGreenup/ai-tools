# AI Tools
CLI in python that uses ollama to provide a basic workflow for search, chat, rag and visualization of docs and notes. I use this primarily for search and as an alternative to [open-webui](https://docs.openwebui.com/), so I don't have to leave my editor.

Read [the documentation](./docs/index.md) for more information.
## Installation

### pipx

```sh
cd $(mktemp -d)
git clone https://github.com/RyanGreenup/ai-tools
cd ai-tools
pipx install .
ai-tools --help
```


### Poetry
Install with poetry:

```sh
git clone https://github.com/RyanGreenup/ai-tools
cd ai-tools
poetry install

# In the directory
poetry run src/main.py -n ~/Notes/slipbox live-search

# Anywhere
dir=$(pwd)
cd /tmp/
poetry run -C "${dir}" /home/ryan/.local/scripts/python/ai-tools/src/main.py  --help
```

Consider creating a script, e.g. =~/.local/bin/ait:

``` sh
#!/bin/sh
dir="$HOME/.local/scripts/python/ai-tools/"
poetry run -C "${dir}" "${dir}"/src/main.py ${@:-}
```

### Virtual environment

```sh
git clone https://github.com/RyanGreenup/ai-tools
cd ai-tools
python -m venv venv
source venv/bin/activate
pip install .
python src/main.py --help

```
## Screenshots

### CLI
![](assets/cli.png)
### Embedding Space
![](assets/semantic_space_plot.png)
### Semantic Search
![](assets/live-search.png)




## TODO

- [x] Chat
- [x] RAG
- [x] Visualization
- [x] Search
    - [x] Live Search
- [ ] RAG generate Answers over question sets to generate answers
- [ ] Question / Answer Generation
- [ ] Documentation summaries using
    - [ ] Map Reduce
    - [ ] Semantic Space Clustering
