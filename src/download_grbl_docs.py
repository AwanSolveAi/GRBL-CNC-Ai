from pathlib import Path
import requests


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "raw" / "grbl"

DOCUMENTS = {
    "settings.md": "https://raw.githubusercontent.com/gnea/grbl/master/doc/markdown/settings.md",
    "commands.md": "https://raw.githubusercontent.com/gnea/grbl/master/doc/markdown/commands.md",
}


def download_file(filename: str, url: str) -> bool:
    output_path = OUTPUT_DIR / filename

    print("=" * 80)
    print(f"Downloading: {filename}")
    print(f"URL: {url}")

    try:
        response = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent": "GRBL-CNC-AI-Knowledge-Builder/1.0"
            },
        )

        response.raise_for_status()

        output_path.write_text(
            response.text,
            encoding="utf-8"
        )

        print(f"Saved: {output_path}")
        print(f"Characters: {len(response.text):,}")
        return True

    except requests.RequestException as exc:
        print(f"DOWNLOAD FAILED: {exc}")
        return False


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print("GRBL CNC AI - OFFICIAL DOCUMENT DOWNLOADER")
    print("=" * 80)
    print()

    successful = 0
    failed = 0

    for filename, url in DOCUMENTS.items():
        if download_file(filename, url):
            successful += 1
        else:
            failed += 1

    print()
    print("=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print()
    print(f"Knowledge base:")
    print(OUTPUT_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()
