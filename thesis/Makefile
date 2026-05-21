PDF=main.pdf
TEX=main.tex

.PHONY: all pdf clean

all: pdf

pdf:
	pdflatex -interaction=nonstopmode $(TEX)
	-bibtex main
	pdflatex -interaction=nonstopmode $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)

clean:
	rm -f *.aux *.bbl *.blg *.lof *.log *.lot *.out *.toc *.fls *.fdb_latexmk *.synctex.gz
