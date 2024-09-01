PYTHON = .venv/bin/python
MATURIN = .venv/bin/maturin

# https://www.maturin.rs/#usage
dev:
	$(MATURIN) develop

check:
	$(PYTHON) simple.py

.PHONY: dev