# run: clean.done
run:
	python setup.py install
	rm clean.done
test:
	clear
	python test.py
clean.done:
	rm -r build
	rm -r dist
	rm -r *.egg-info
	rm -r ~/.conda/envs/torch/lib/python3.7/site-packages/lptorch-0.0.0-py3.7-linux-x86_64.egg
	touch clean.done
clean: clean.done
