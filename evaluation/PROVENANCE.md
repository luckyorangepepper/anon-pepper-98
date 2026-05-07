# Evaluation Provenance

These rubrics match the scoring prompts used for the MATRIX experiments. The
only mechanical change is that single-example image-caption scoring is converted
to the same batch JSON-list format used by the text scorer.

Prompt mapping from the original internal experiment scorers:

- `eval/general_score_gpt.py::prompt_scoring` -> `rubrics/text_qa.md`
- `eval/general_score_gpt.py::hypothesis_prompt_scoring` -> `rubrics/hypothesis_generation.md`
- `captions/eval_llm_judge.py::PROMPT_TEMPLATE` -> `rubrics/image_caption.md`

The original image-caption scorer used a different API client. This artifact
keeps the same image-caption scoring rubric but routes requests through the
paper judge default, `claude-opus-4-5`.

Accepted input schemas:

- Text and hypothesis rows: `question`, `answer`, `model_answer`
- Image-caption rows: `reference_caption`, `predicted_caption`, optional
  `paper_id`, `image_id`, `kind`, and `image_path`
