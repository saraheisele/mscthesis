# Tell latexmk to put all aux and output files in 'build'
$out_dir = 'build';
$aux_dir = 'build';

# Make PDF output by default
$pdf_mode = 1;

# Stop if there are serious errors
$failure_cmd = 'echo "❌ LaTeX build failed."';

# Optional: use synctex for source<->pdf linking
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';
