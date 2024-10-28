import datetime


def currentDateTime() -> str:
    """
    Get the current date and time in a formatted string.

    Returns:
        str: Current datetime in format 'YYYY-MM-DD HH:MM:SS'
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
