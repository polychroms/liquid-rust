#!/usr/bin/env bash
# Liquid-Rust system-wide installer (Linux x86_64 / ARM64 & macOS)
# Works both standalone (downloads latest release tarball from GitHub) and offline
# inside an extracted release archive containing `liquid-rust` + `liquid-rust-install.sh`.

set -euo pipefail

REPO="rappar222/liquid-rust"

BOLD="\033[1m"
GREEN="\033[32m"
CYAN="\033[36m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

FORCE_DOWNLOAD=0
for arg in "$@"; do
    case "$arg" in
        --download|--online)
            FORCE_DOWNLOAD=1
            ;;
        -h|--help)
            echo "Usage: $0 [--download]"
            echo "  Installs liquid-rust to /bin (Linux) or /usr/local/bin (macOS)."
            echo "  When run inside an extracted release folder alongside 'liquid-rust', installs locally."
            echo "  Use --download (or curl ... | bash) to fetch the latest release from GitHub."
            exit 0
            ;;
    esac
done

echo -e "${CYAN}${BOLD}"
echo "============================================================"
echo "          🌊 Liquid-Rust 3D/2D Fluid Simulator              "
echo "             System-Wide Installer                          "
echo "============================================================"
echo -e "${RESET}"

# 1. Detect OS and CPU architecture
OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"

case "$OS_NAME" in
    Linux*)
        case "$ARCH_NAME" in
            x86_64|amd64)
                ASSET_NAME="liquid-rust-linux-x86_64.tar.gz"
                ;;
            aarch64|arm64)
                ASSET_NAME="liquid-rust-linux-aarch64.tar.gz"
                ;;
            *)
                echo -e "${RED}Error: Unsupported Linux architecture '${ARCH_NAME}'.${RESET}" >&2
                exit 1
                ;;
        esac
        TARGET_DIR="/bin"
        ;;
    Darwin*)
        case "$ARCH_NAME" in
            arm64|aarch64)
                ASSET_NAME="liquid-rust-macos-arm64.tar.gz"
                ;;
            x86_64)
                ASSET_NAME="liquid-rust-macos-x86_64.tar.gz"
                ;;
            *)
                echo -e "${RED}Error: Unsupported macOS architecture '${ARCH_NAME}'.${RESET}" >&2
                exit 1
                ;;
        esac
        # macOS SIP locks /bin; /usr/local/bin is in default global PATH
        if [ -w "/bin" ]; then
            TARGET_DIR="/bin"
        else
            TARGET_DIR="/usr/local/bin"
        fi
        ;;
    *)
        echo -e "${RED}Error: Unsupported operating system '${OS_NAME}'.${RESET}" >&2
        exit 1
        ;;
esac

TMP_DIR="$(mktemp -d)"
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# 2. Use bundled binary if invoked directly from an extracted tarball directory
SCRIPT_DIR=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

SOURCE_BIN=""
if [ "$FORCE_DOWNLOAD" -eq 0 ] && [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/liquid-rust" ]; then
    BIN_SIZE="$(wc -c < "$SCRIPT_DIR/liquid-rust" | tr -d '[:space:]')"
    if [ "${BIN_SIZE:-0}" -ge 65536 ]; then
        SOURCE_BIN="$SCRIPT_DIR/liquid-rust"
        echo -e "${CYAN}⚙ Found extracted binary (${SOURCE_BIN}), installing locally...${RESET}"
    fi
fi

# 3. Otherwise download and extract latest release tarball from GitHub
if [ -z "$SOURCE_BIN" ]; then
    DOWNLOAD_URL="https://github.com/${REPO}/releases/latest/download/${ASSET_NAME}"
    echo -e "${YELLOW}⬇ Downloading latest package (${ASSET_NAME}) from GitHub...${RESET}"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --progress-bar "$DOWNLOAD_URL" -o "$TMP_DIR/$ASSET_NAME"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --show-progress "$DOWNLOAD_URL" -O "$TMP_DIR/$ASSET_NAME"
    else
        echo -e "${RED}Error: Neither 'curl' nor 'wget' is installed.${RESET}" >&2
        exit 1
    fi

    echo -e "${CYAN}⚙ Extracting package...${RESET}"
    tar -xzf "$TMP_DIR/$ASSET_NAME" -C "$TMP_DIR"

    if [ ! -f "$TMP_DIR/liquid-rust" ]; then
        echo -e "${RED}Error: Extracted archive did not contain 'liquid-rust' binary.${RESET}" >&2
        exit 1
    fi
    SOURCE_BIN="$TMP_DIR/liquid-rust"
fi

chmod 0755 "$SOURCE_BIN"

# 4. Install binary atomically into system bin directory
TARGET_BIN="${TARGET_DIR}/liquid-rust"
SUDO=""
if [ "${EUID:-$(id -u)}" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
        echo -e "${YELLOW}🔒 Root permission required to install into ${TARGET_BIN}:${RESET}"
    else
        echo -e "${RED}Error: Must run as root (or have 'sudo' installed) to write to ${TARGET_BIN}.${RESET}" >&2
        exit 1
    fi
fi

$SUDO mkdir -p "$TARGET_DIR"
$SUDO install -m 0755 "$SOURCE_BIN" "${TARGET_BIN}.new"
$SUDO mv -f "${TARGET_BIN}.new" "$TARGET_BIN"

# 5. Install desktop application entry on Linux
if [ "$OS_NAME" = "Linux" ]; then
    DESKTOP_TMP="$TMP_DIR/liquid-rust.desktop"
    cat <<EOF > "$DESKTOP_TMP"
[Desktop Entry]
Name=Liquid-Rust
Comment=High-Performance 3D/2D SPH Fluid Simulator (GPU Accelerated)
Exec=${TARGET_BIN}
Icon=liquid-rust
Terminal=false
Type=Application
Categories=Graphics;Science;Simulation;
EOF
    if [ -d "/usr/share/applications" ]; then
        $SUDO install -m 0644 "$DESKTOP_TMP" "/usr/share/applications/liquid-rust.desktop" || true
    fi
fi

INSTALLED_VER="$("$TARGET_BIN" --version 2>/dev/null || echo "liquid-rust")"

echo ""
echo -e "${GREEN}${BOLD}✔ Installed ${INSTALLED_VER} to ${TARGET_BIN}!${RESET}"
echo ""
echo -e "${BOLD}Run from anywhere:${RESET}"
echo -e "  ${CYAN}liquid-rust${RESET}                      # Launch interactive 3D/2D Studio GUI"
echo -e "  ${CYAN}liquid-rust --check-update${RESET}       # Check GitHub for updates"
echo -e "  ${CYAN}liquid-rust --update${RESET}             # Download & install latest update"
echo -e "  ${CYAN}liquid-rust --help${RESET}               # View all CLI & MP4 export flags"
