"""The ORIGINAL script, as the assessment describes it: "fails on complex pages or long content".

The brief did not ship the script, so this is my hypothetical version of the common way it is
written. It is kept unchanged so the fix can be compared with it (`python -m aac scrape --naive`).
Each `# PROBLEM` comment is one failure mode; the README (Part 2) explains them, with evidence in docs/evidence/.
"""

import requests
from bs4 import BeautifulSoup


def scrape_and_summarize(url, llm):
    html = requests.get(url).text  # PROBLEM 1: no timeout, no retry, no status check, no size limit

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()  # PROBLEM 2: keeps <script>/<style> JSON, nav, footer, cookie banners
    # PROBLEM 3: a JavaScript-rendered page returns an empty shell here, and nothing notices

    prompt = "Summarize this webpage:\n\n" + text  # PROBLEM 4 (the bottleneck): the whole page in ONE prompt.
    # Cost and latency grow with page size; past the context window the provider either errors
    # or (Ollama, num_ctx) silently truncates the prompt, so the model may never see the instruction
    # or most of the page.

    return llm.invoke(prompt).content  # PROBLEM 5: no length limit or check on the summary
