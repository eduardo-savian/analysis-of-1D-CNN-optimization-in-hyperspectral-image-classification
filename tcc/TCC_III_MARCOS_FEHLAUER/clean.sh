
set -eu

TrashFiles(){
	find .  -name '*.aux' \
		-or -name '*.log' \
		-or -name '*.fls' \
		-or -name '*.loq' \
		-or -name '*.loe' \
		-or -name '*.bbl' \
		-or -name '*.blg' \
		-or -name '*.lof' \
		-or -name '*.toc' \
		-or -name '*.dvi' \
		-or -name '*.ps' \
		-or -name '*.fdb_*'
}

files="$(TrashFiles)"

for f in $files; do
	echo "Delete $f"
	rm -f "$f"
done

