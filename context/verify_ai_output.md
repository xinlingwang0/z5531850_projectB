# Verify AI Output Before You Use It (provided - do not edit)

Read this before you let any AI-written text, number, or citation into your report,
your code, or your app. It exists because in 2026 two of the world's largest firms,
KPMG and EY, published reports that were full of AI hallucinations - fake citations,
invented case studies, and statistics that did not exist. Both were retracted and
made headline news. Do not repeat their mistake.

## The rule

Treat every AI output as a draft to be checked, not a fact to be trusted.

## Before anything goes into your report, code, or app, confirm:

- Every citation points to a real source you have opened yourself. If you cannot
  find it, delete it. Do not keep a reference you have not seen.
- Every number traces to the data or to a computation you can re-run. No number
  goes in because "the AI said so".
- Every factual claim about a company, dataset, or method is something you can
  support. If you cannot, cut it or mark it clearly as uncertain.
- Code does what you think it does. You have run it and checked the output, not
  just read it.

## Make your AI agent enforce this

Put rules like these in your own agent or instruction files (AGENTS.md, CLAUDE.md,
or `.claude/`), so the assistant is told to:

- never invent a citation, a statistic, or a source,
- flag any claim it cannot verify instead of stating it confidently,
- show its working for any number it produces,
- and remind you to check its output before you use it.

Doing this well is exactly what the AI Workflow mark rewards - you direct the tool,
then catch and correct what it gets wrong.
