Given a ground-truth caption and a model-generated caption for the same image, rate how semantically similar they are.
A high score should indicate that the model-generated caption accurately preserves the essential content of the ground-truth caption, including objects, attributes, actions, spatial relationships, and notable visual details, even if the wording differs.

Ground-truth caption:
{reference}

Model caption:
{prediction}

Scoring rubric (5-tier):
- 1.0: Excellent - Semantically equivalent or very close; conveys the same meaning and all key details (phrasing differences are fine).
- 0.75: Good - Captures most key elements with only minor omissions or small inaccuracies that don't change the overall meaning.
- 0.5: Partial - Captures some key elements but misses important details or has notable inaccuracies.
- 0.25: Poor - Gets a few things right but misses most key information or has significant errors.
- 0.0: Incorrect - Mostly incorrect, barely related, or completely wrong.

Respond ONLY with a JSON object like {{"score": 0.5, "explanation": "short reason"}} where score is 0, 0.25, 0.5, 0.75, or 1.
Keep the explanation concise (one sentence).
