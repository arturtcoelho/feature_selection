PDF=main.pdf
TEX=main.tex

.PHONY: all pdf clean anonymous

all: pdf

pdf:
	pdflatex -interaction=nonstopmode $(TEX)
	-bibtex main
	pdflatex -interaction=nonstopmode $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)

anonymous:
	perl -0777 -pe 's/portada\.pdf/portada_anonym.pdf/g; s/\\begin\{center\}\s*\\LARGE\s*\\textbf\{Acknowledgements\}.*?\\newpage\n//s' $(TEX) > main_anon.tex
	pdflatex -interaction=nonstopmode main_anon.tex
	-bibtex main_anon
	pdflatex -interaction=nonstopmode main_anon.tex
	pdflatex -interaction=nonstopmode main_anon.tex
	mv main_anon.pdf main_anonymous.pdf
	rm -f main_anon.*

clean:
	rm -f *.aux *.bbl *.blg *.lof *.log *.lot *.out *.toc *.fls *.fdb_latexmk *.synctex.gz
