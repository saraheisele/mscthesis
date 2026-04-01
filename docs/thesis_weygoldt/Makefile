.PHONY: clean

THESIS=Thesis

build:
	latexmk -bibtex -pdf -f -pdflatex="pdflatex -synctex=1 -interaction=nonstopmode" -xelatex $(THESIS).tex

clean:
	latexmk -c

watch:
	latexmk -quiet -bibtex -pvc -f -pdf -pdflatex="pdflatex -synctex=1 -interaction=nonstopmode" -xelatex $(THESIS).tex
