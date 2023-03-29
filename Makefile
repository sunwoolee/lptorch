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
	touch clean.done
clean: clean.done
