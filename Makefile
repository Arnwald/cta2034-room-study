.PHONY: all test study coherent library slopes figures articles paper clean

all: study coherent library slopes figures paper

test:
	cargo test --release

study:
	cargo run --release --bin study

coherent:
	cargo run --release --bin coherent

library:
	cargo run --release --bin library
	cargo run --release --bin standard

slopes:
	cargo run --release --bin slopes -- run
	cargo run --release --bin slopes -- tex

figures:
	cargo run --release --bin figures

articles:
	cargo run --release --bin articles

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex main_fr.tex

clean:
	cargo clean
	cd paper && latexmk -C main.tex main_fr.tex
