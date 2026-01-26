#!/usr/bin/env python
import os
from argostranslate import package


def _get_data_dir() -> str:
    return (
        os.getenv("ARGOS_TRANSLATE_DATA_DIR")
        or os.getenv("XDG_DATA_HOME")
        or os.path.expanduser("~/.local/share/argos-translate")
    )


def main() -> None:
    data_dir = _get_data_dir()
    marker = os.path.join(data_dir, ".all_installed")
    if os.path.exists(marker):
        print("Argos packages already installed; skipping.")
        return

    os.makedirs(data_dir, exist_ok=True)
    print("Updating Argos package index...")
    package.update_package_index()
    packages = package.get_available_packages()
    print(f"Available packages: {len(packages)}")

    failures = 0
    for pkg in packages:
        try:
            path = pkg.download()
            package.install_from_path(path)
            print(f"Installed {pkg.from_code}->{pkg.to_code}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"Failed {pkg.from_code}->{pkg.to_code}: {exc}")

    if failures == 0:
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write("ok\n")
        print("All Argos packages installed.")
    else:
        print(f"Argos install finished with {failures} failures; marker not written.")


if __name__ == "__main__":
    main()
