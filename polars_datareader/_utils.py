"""Utility functions and classes for polars-datareader."""

import datetime

import httpx


class RemoteDataError(IOError):
    """
    Exception raised when remote data retrieval fails.

    This exception is raised when data cannot be fetched from a remote
    source after all retry attempts have been exhausted.
    """

    pass


class SymbolWarning(UserWarning):
    """
    Warning for symbol-related issues.

    This warning is raised when a specific symbol fails to load
    but processing continues with other symbols.
    """

    pass


def _sanitize_dates(
    start: str | datetime.date | datetime.datetime | int | None,
    end: str | datetime.date | datetime.datetime | int | None,
) -> tuple[datetime.datetime, datetime.datetime]:
    """
    Sanitize and normalize start and end date parameters.

    Parameters
    ----------
    start : str, date, datetime, int, or None
        Start date. If int, treated as year (e.g., 2020 -> 2020-01-01).
        If None, defaults to 5 years before today.
    end : str, date, datetime, int, or None
        End date. If int, treated as year (e.g., 2020 -> 2020-01-01).
        If None, defaults to today.

    Returns
    -------
    tuple[datetime.datetime, datetime.datetime]
        Normalized (start, end) as datetime objects.

    Raises
    ------
    ValueError
        If start date is after end date or dates are malformed.

    Examples
    --------
    >>> _sanitize_dates(2020, 2021)
    (datetime.datetime(2020, 1, 1, 0, 0), datetime.datetime(2021, 1, 1, 0, 0))

    >>> _sanitize_dates("2020-01-01", "2021-12-31")
    (datetime.datetime(2020, 1, 1, 0, 0), datetime.datetime(2021, 12, 31, 0, 0))
    """
    # Default start: 5 years ago
    if start is None:
        today = datetime.date.today()
        start = today - datetime.timedelta(days=365 * 5)

    # Default end: today
    if end is None:
        end = datetime.date.today()

    # Convert integers to datetime (treat as year)
    if isinstance(start, int):
        start = datetime.datetime(start, 1, 1)

    if isinstance(end, int):
        end = datetime.datetime(end, 1, 1)

    # Convert date to datetime
    if isinstance(start, datetime.date) and not isinstance(start, datetime.datetime):
        start = datetime.datetime.combine(start, datetime.time.min)

    if isinstance(end, datetime.date) and not isinstance(end, datetime.datetime):
        end = datetime.datetime.combine(end, datetime.time.min)

    # Convert strings to datetime
    if isinstance(start, str):
        try:
            # Try ISO format first
            start = datetime.datetime.fromisoformat(start)
        except ValueError:
            # Try common formats
            for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%Y%m%d"]:
                try:
                    start = datetime.datetime.strptime(start, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Unable to parse start date: {start}")

    if isinstance(end, str):
        try:
            # Try ISO format first
            end = datetime.datetime.fromisoformat(end)
        except ValueError:
            # Try common formats
            for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%Y%m%d"]:
                try:
                    end = datetime.datetime.strptime(end, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Unable to parse end date: {end}")

    # Validate
    if start > end:
        raise ValueError(
            f"Start date ({start}) must be before or equal to end date ({end})"
        )

    return start, end


def _init_session(
    session: httpx.Client | None,
    client_class: type[httpx.Client] = httpx.Client,
) -> httpx.Client:
    """
    Initialize or validate an HTTP client session.

    Parameters
    ----------
    session : httpx.Client or None
        Existing session to validate, or None to create new one.
    client_class : type[httpx.Client], default httpx.Client
        Client class to instantiate if session is None.

    Returns
    -------
    httpx.Client
        Valid HTTP client session.

    Raises
    ------
    TypeError
        If session is provided but not an instance of httpx.Client.

    Examples
    --------
    >>> session = _init_session(None)
    >>> isinstance(session, httpx.Client)
    True

    >>> custom_session = httpx.Client(timeout=60.0)
    >>> session = _init_session(custom_session)
    >>> session is custom_session
    True
    """
    if session is None:
        return client_class()

    if not isinstance(session, httpx.Client):
        raise TypeError(
            f"session must be an httpx.Client instance, got {type(session)}"
        )

    return session
