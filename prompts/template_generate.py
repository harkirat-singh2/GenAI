from langchain_core.prompts import PromptTemplate

template = """
Please summarize the research paper titled "{paper_name}" with the following specifications:

Explanation Style: {style}
Explanation Length: {length}

1. Mathematical Details:
- Include relevant mathematical equations if present in the paper.
- Explain mathematical concepts using intuitive code snippets.

2. Analogies:
- Use relatable analogies.

If information is unavailable, say:
"Insufficient information available"

Ensure clarity and accuracy.
"""

prompt = PromptTemplate(
    input_variables=["paper_name", "style", "length"],
    validate_template=True,
    template=template,
)

# ✅ Save correctly
prompt.save("template.json")
