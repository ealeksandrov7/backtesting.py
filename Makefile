VENV_PYTHON ?= .venv/bin/python

.PHONY: check-trading-env install-trading ui bot bot-once

check-trading-env:
	@test -x "$(VENV_PYTHON)" || (echo "Missing $(VENV_PYTHON). Create the repo venv first." && exit 1)

install-trading: check-trading-env
	$(VENV_PYTHON) -m pip install setuptools wheel streamlit bokeh hyperliquid-python-sdk eth-account pydantic numpy pandas

ui: check-trading-env
	$(VENV_PYTHON) -m streamlit run trading/streamlit_app.py

bot: check-trading-env
	$(VENV_PYTHON) -m trading.bot $(ARGS)

bot-once: check-trading-env
	$(VENV_PYTHON) -m trading.bot --once $(ARGS)
