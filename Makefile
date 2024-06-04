ui:
	streamlit run ui.py
run: 
	./generate.py --config ./b_on/generate.cfg
	./simulate.py --config ./b_on/simulate.cfg b_on/autogen/scenario.json