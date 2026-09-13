# MLX-Serve fork

## Upstream integration quality

- Use every upstream sync to reduce fork divergence; prioritize correctness and long-term maintainability over execution speed.
- Classify each substantial customization as retired, adapted to upstream, or retained; record its disposition and evidence in the merge review.
- Prefer upstream implementations and narrow extension points. Remove superseded code and duplicate paths; isolate necessary overrides from shared code.
- Preserve required behavior and regression coverage. Prove equivalence before retiring a customization; passing tests alone does not establish upstream parity.
- Review architecture, future merge burden, and behavioral correctness before landing. Do not minimize conflicts by blindly retaining either side.

## Working rules

- Merge upstream history; do not rebase published fork history.
- Test affected behavior and preserve registry-to-worker configuration contracts.
- Keep agent-facing instructions terse; put rationale and measurements in human-facing documentation.
- Commit coherent work. Push only on explicit approval in the current turn.
