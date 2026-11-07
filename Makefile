.PHONY: all test study coherent figures paper clean

all: study coherent figures paper

test:
	cargo test --release

study:
	cargo run --release --bin study

coherent:
	cargo run --release --bin coherent

figures:
	cargo run --release --bin figures

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex main_fr.tex

clean:
	cargo clean
	cd paper && latexmk -C main.tex main_fr.tex
