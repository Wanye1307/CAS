The bundled `src/medvlm` package was copied from the local MedVLM checkout
whose LICENSE names Young Joon Park and grants permission under the MIT
License. Its original project URL was
https://github.com/JiyoungByun/medvlm . The local checkout contained
uncommitted model and confidence changes. This release preserves the
upstream LICENSE and adapts model paths through `CAS_MODEL_ROOT`.

CAS benchmark scripts, `confidence/grounding.py`, and `evaluation/selective.py`
were collected from the same local research workspace. Publication of this
package should be reviewed by the project owners for authorship and dataset
rights. No dataset or model weight files are included.
