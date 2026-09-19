# Most LLM Conformity Needs No Speaker
### Measuring the Speaker-Free Floor in Peer-Pressure Benchmarks

Yibo Hu (Illinois Institute of Technology) · Jiaming Qu (Amazon)

📄 Paper: [arXiv:2607.05545](https://arxiv.org/abs/2607.05545)

## Key finding

Most of what looks like LLM conformity survives even after the speaker is removed.
A standard conformity prompt mixes two cues at once — an explicit speaker and a repeated
wrong answer. Holding the asserted answer fixed and deleting the speaker isolates how much
revision actually depends on the speaker.

Across six open-weight LLMs and seven QA and reasoning datasets:

- A **no-source** assertion ("The answer is X.") alone drives harmful revision to **66.5%**,
  versus **10.3%** for a plain re-ask.
- **Expert-panel** framing adds a robust increment on top (**79.4%**, +12.9 pp); a bare person
  label does not reliably add anything (**57.4%**).

> Source attribution still matters, but it should be measured as an increment above a large
> speaker-free floor, not as the whole effect.

## Code and data

The reproduction package — the prompt-perturbation construction, the deterministic
log-probability arbitration read, the revision metrics, an example input, and the
mechanism-contrast table behind the headline numbers — **will be released here upon
publication. In the meantime, it is available from the authors on request.**

## Citation

```bibtex
@article{hu2026most,
  title  = {Most LLM Conformity Needs No Speaker: Measuring the Speaker-Free Floor in Peer-Pressure Benchmarks},
  author = {Hu, Yibo and Qu, Jiaming},
  journal= {arXiv preprint arXiv:2607.05545},
  year   = {2026},
}
```
