#!/bin/sh
set -eu

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg is not available; GIF products remain the accepted movies." >&2
    exit 1
fi

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
for representation in A B C; do
    for method in M1Y H1 H2 H5; do
        source="$root/movies/representation_${representation}/movie_rep${representation}_${method}.gif"
        destination="$root/movies/representation_${representation}/movie_rep${representation}_${method}.mp4"
        if [ -e "$destination" ]; then
            echo "refusing to overwrite $destination" >&2
            exit 1
        fi
        ffmpeg -i "$source" -movflags faststart -pix_fmt yuv420p "$destination"
    done
done
