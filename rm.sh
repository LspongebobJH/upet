dirs=(
    outputs/2026-06-03 outputs/2026-06-04 outputs/2026-06-08 outputs/2026-06-09 outputs/2026-06-10
    outputs/2026-06-15/15-05-27 outputs/2026-06-15/19-48-52 outputs/2026-06-17 outputs/2026-06-22
)

for dir in "${dirs[@]}"; do
    sudo rm -rf "$dir"
done