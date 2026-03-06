"""CLI utilities for consistent output formatting."""

import sys
from typing import Optional, TextIO


class Colors:
    """ANSI color codes for terminal output."""

    # Reset
    RESET = "\033[0m"

    # Text colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"


class CLIOutput:
    """CLI output formatter with color support and verbosity control."""

    def __init__(
        self, quiet: bool = False, verbose: bool = False, use_colors: Optional[bool] = None
    ):
        """Initialize CLI output formatter.

        Args:
            quiet: If True, suppress non-essential output
            verbose: If True, show verbose output
            use_colors: If True/False, force color on/off. If None, auto-detect.
        """
        self.quiet = quiet
        self.verbose = verbose

        # Auto-detect color support if not explicitly set
        if use_colors is None:
            # Check if stdout is a TTY and supports colors
            self.use_colors = sys.stdout.isatty() and hasattr(sys.stdout, "isatty")
            # On Windows, check for colorama or similar
            if sys.platform == "win32":
                try:
                    import colorama

                    colorama.init()
                    self.use_colors = True
                except ImportError:
                    self.use_colors = False
        else:
            self.use_colors = use_colors

    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled.

        Args:
            text: Text to colorize
            color: Color code

        Returns:
            Colorized text or original text if colors disabled
        """
        if self.use_colors:
            return f"{color}{text}{Colors.RESET}"
        return text

    def success(self, message: str) -> None:
        """Print success message.

        Args:
            message: Success message to print
        """
        if not self.quiet:
            print(self._colorize(f"✓ {message}", Colors.GREEN), file=sys.stdout)

    def error(self, message: str, details: Optional[str] = None) -> None:
        """Print error message.

        Args:
            message: Error message to print
            details: Optional additional details
        """
        print(self._colorize(f"✗ Error: {message}", Colors.RED), file=sys.stderr)
        if details and self.verbose:
            print(self._colorize(f"  {details}", Colors.DIM), file=sys.stderr)

    def warning(self, message: str) -> None:
        """Print warning message.

        Args:
            message: Warning message to print
        """
        if not self.quiet:
            print(self._colorize(f"⚠  {message}", Colors.YELLOW), file=sys.stderr)

    def info(self, message: str) -> None:
        """Print info message.

        Args:
            message: Info message to print
        """
        if not self.quiet:
            print(message, file=sys.stdout)

    def verbose_msg(self, message: str) -> None:
        """Print verbose message (only if verbose mode enabled).

        Args:
            message: Verbose message to print
        """
        if self.verbose and not self.quiet:
            print(self._colorize(f"  {message}", Colors.DIM), file=sys.stdout)

    def header(self, message: str) -> None:
        """Print header message.

        Args:
            message: Header message to print
        """
        if not self.quiet:
            print(self._colorize(f"\n{message}", Colors.BOLD + Colors.CYAN), file=sys.stdout)

    def section(self, message: str) -> None:
        """Print section message.

        Args:
            message: Section message to print
        """
        if not self.quiet:
            print(self._colorize(f"  {message}", Colors.CYAN), file=sys.stdout)

    def print(self, message: str, file: Optional[TextIO] = None) -> None:
        """Print raw message.

        Args:
            message: Message to print
            file: File to print to (default: stdout)
        """
        if not self.quiet:
            print(message, file=file or sys.stdout)

    def progress(self, message: str) -> None:
        """Print progress message.

        Args:
            message: Progress message to print
        """
        if not self.quiet:
            print(self._colorize(f"→ {message}", Colors.BLUE), file=sys.stdout)

    def result(self, message: str) -> None:
        """Print result message.

        Args:
            message: Result message to print
        """
        if not self.quiet:
            print(self._colorize(message, Colors.BRIGHT_GREEN), file=sys.stdout)


# Global CLI output instance (will be initialized in main())
_output: Optional[CLIOutput] = None


def init_output(
    quiet: bool = False, verbose: bool = False, use_colors: Optional[bool] = None
) -> None:
    """Initialize global CLI output instance.

    Args:
        quiet: If True, suppress non-essential output
        verbose: If True, show verbose output
        use_colors: If True/False, force color on/off. If None, auto-detect.
    """
    global _output
    _output = CLIOutput(quiet=quiet, verbose=verbose, use_colors=use_colors)


def get_output() -> CLIOutput:
    """Get global CLI output instance.

    Returns:
        CLIOutput instance

    Raises:
        RuntimeError: If output not initialized
    """
    global _output
    if _output is None:
        # Default initialization if not explicitly initialized
        _output = CLIOutput()
    return _output
