"""Base classes for data readers."""

import datetime
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from io import StringIO
from urllib.parse import urlencode

import httpx
import polars as pl

from polars_datareader._utils import (
    RemoteDataError,
    SymbolWarning,
    _sanitize_dates,
)


class BaseReader(ABC):
    """
    Abstract base class for data readers.

    This class provides the core functionality for fetching data from remote
    sources with retry logic, session management, and error handling. Subclasses
    must implement the `url` property and optionally override other methods to
    customize behavior.

    Parameters
    ----------
    symbols : str, int, Sequence[str], or pl.DataFrame
        Symbol(s) to fetch data for. Can be a single symbol or multiple.
    start : str, date, datetime, int, or None, default None
        Start date for data retrieval. If None, defaults to 5 years ago.
    end : str, date, datetime, int, or None, default None
        End date for data retrieval. If None, defaults to today.
    retry_count : int, default 3
        Number of retry attempts for failed requests.
    pause : float, default 0.1
        Initial pause duration (in seconds) between retries.
    timeout : float, default 30.0
        Request timeout in seconds.
    session : httpx.Client or None, default None
        HTTP client session. If None, a new session is created.
    freq : str or None, default None
        Data frequency (e.g., 'daily', 'weekly'). Implementation-specific.

    Attributes
    ----------
    symbols : str, int, Sequence[str], or pl.DataFrame
        Symbols to fetch.
    start : datetime.datetime
        Normalized start date.
    end : datetime.datetime
        Normalized end date.
    retry_count : int
        Number of retry attempts.
    pause : float
        Pause duration between retries.
    timeout : float
        Request timeout.
    pause_multiplier : int
        Multiplier for exponential backoff (default: 1, no backoff).
    session : httpx.Client
        HTTP client session.
    freq : str or None
        Data frequency.
    headers : dict or None
        Custom HTTP headers.

    Examples
    --------
    >>> class MyReader(BaseReader):
    ...     @property
    ...     def url(self) -> str:
    ...         return "https://api.example.com/data"
    ...
    >>> with MyReader("AAPL") as reader:
    ...     data = reader.read()
    """

    _chunk_size: int = 1024 * 1024
    _format: str = "string"

    def __init__(
        self,
        symbols: str | int | Sequence[str] | pl.DataFrame,
        start: str | datetime.date | datetime.datetime | int | None = None,
        end: str | datetime.date | datetime.datetime | int | None = None,
        retry_count: int = 3,
        pause: float = 0.1,
        timeout: float = 30.0,
        session: httpx.Client | None = None,
        freq: str | None = None,
    ) -> None:
        self.symbols = symbols
        self.start, self.end = _sanitize_dates(
            start or self.default_start_date, end
        )

        if not isinstance(retry_count, int) or retry_count < 0:
            raise ValueError("'retry_count' must be integer larger than 0")

        self.retry_count = retry_count
        self.pause = pause
        self.timeout = timeout
        self.pause_multiplier = 1
        self.session = session or httpx.Client()
        self.freq = freq
        self.headers: dict[str, str] | None = None


    @property
    def default_start_date(self) -> datetime.date:
        """
        Default start date for data retrieval (5 years ago).

        Returns
        -------
        datetime.date
            Date from 5 years ago.
        """
        today = datetime.date.today()
        return today - datetime.timedelta(days=365 * 5)

    @property
    @abstractmethod
    def url(self) -> str:
        """
        API URL for data retrieval.

        This property must be implemented by subclasses to specify the
        endpoint URL for fetching data.

        Returns
        -------
        str
            API endpoint URL.

        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        """
        raise NotImplementedError("Subclass must implement 'url' property")

    @property
    def params(self) -> dict[str, str | int | float] | None:
        """
        Parameters to use in API calls.

        Subclasses can override this to provide query parameters.

        Returns
        -------
        dict or None
            Query parameters for the API request.
        """
        return None

    def __enter__(self) -> "BaseReader":
        """
        Enter context manager.

        Returns
        -------
        BaseReader
            Self for use in with statement.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """
        Exit context manager and close session.

        Parameters
        ----------
        exc_type : type[BaseException] or None
            Exception type if an exception occurred.
        exc_val : BaseException or None
            Exception value if an exception occurred.
        exc_tb : object
            Exception traceback if an exception occurred.
        """
        self.close()

    def read(self) -> pl.DataFrame:
        """
        Read data from connector.

        Returns
        -------
        pl.DataFrame
            Retrieved data.

        Raises
        ------
        RemoteDataError
            If data retrieval fails.
        OSError
            If response is empty or invalid.
        """
        try:
            return self._read_one_data(self.url, self.params)
        finally:
            self.close()

    def _read_one_data(
        self, url: str, params: dict[str, str | int | float] | None
    ) -> pl.DataFrame:
        """
        Read one data from specified URL.

        Parameters
        ----------
        url : str
            URL to fetch data from.
        params : dict or None
            Query parameters.

        Returns
        -------
        pl.DataFrame
            Fetched and parsed data.

        Raises
        ------
        NotImplementedError
            If format is not supported.
        """
        if self._format == "string":
            out = self._read_url_as_StringIO(url, params=params)
        elif self._format == "json":
            out = self._get_response(url, params=params).json()
        else:
            raise NotImplementedError(self._format)
        return self._read_lines(out)

    def _read_url_as_StringIO(
        self, url: str, params: dict[str, str | int | float] | None = None
    ) -> StringIO:
        """
        Fetch URL and return content as StringIO buffer.

        Parameters
        ----------
        url : str
            URL to fetch.
        params : dict or None
            Query parameters.

        Returns
        -------
        StringIO
            Response content as string buffer.

        Raises
        ------
        OSError
            If response is empty.
        """
        response = self._get_response(url, params=params)
        text = self._sanitize_response(response)
        out = StringIO()

        if len(text) == 0:
            service = self.__class__.__name__
            raise OSError(
                f"{service} request returned no data; "
                f"check URL for invalid inputs: {self.url}"
            )

        if isinstance(text, bytes):
            out.write(text.decode("utf-8"))
        else:
            out.write(text)

        out.seek(0)
        return out

    @staticmethod
    def _sanitize_response(response: httpx.Response) -> bytes:
        """
        Hook to allow subclasses to clean up response data.

        Parameters
        ----------
        response : httpx.Response
            HTTP response object.

        Returns
        -------
        bytes
            Response content.
        """
        return response.content

    def _get_response(
        self,
        url: str,
        params: dict[str, str | int | float] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """
        Send HTTP GET request with retry logic.

        Parameters
        ----------
        url : str
            Target URL.
        params : dict or None
            Query parameters.
        headers : dict or None
            HTTP headers. If None, uses self.headers.

        Returns
        -------
        httpx.Response
            HTTP response object.

        Raises
        ------
        RemoteDataError
            If all retry attempts fail.
        """
        headers = headers or self.headers
        pause = self.pause
        last_response_text = ""

        for _ in range(self.retry_count + 1):
            response = self.session.get(
                url, params=params, headers=headers, timeout=self.timeout
            )

            # FIX: Use httpx.codes.OK instead of requests.codes.ok
            if response.status_code == httpx.codes.OK:
                return response

            if response.encoding:
                last_response_text = response.text.encode(response.encoding)

            time.sleep(pause)

            # Increase time between subsequent requests, per subclass.
            pause *= self.pause_multiplier

            # Get a new breadcrumb if necessary, in case ours is invalidated
            # FIX: params should be dict, not list
            if isinstance(params, dict) and "crumb" in params:
                params["crumb"] = self._get_crumb(self.retry_count)

            # If our output error function returns True, exit the loop.
            if self._output_error(response):
                break

        if params is not None and len(params) > 0:
            url = url + "?" + urlencode(params)

        msg = f"Unable to read URL: {url}"
        if last_response_text:
            msg += f"\nResponse Text:\n{last_response_text}"

        raise RemoteDataError(msg)

    def _get_crumb(self, *args) -> str:
        """
        Get a new crumb/token for authentication.

        This method should be implemented by subclasses that require
        dynamic authentication tokens.

        Parameters
        ----------
        *args
            Implementation-specific arguments.

        Returns
        -------
        str
            Authentication crumb/token.

        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        """
        raise NotImplementedError("Subclass has not implemented method.")

    def _output_error(self, out: httpx.Response) -> bool:
        """
        Interpret non-200 HTTP responses.

        If necessary, a service can implement an interpreter for any non-200
        HTTP responses to determine if retry should be aborted.

        Parameters
        ----------
        out : httpx.Response
            The HTTP response object.

        Returns
        -------
        bool
            True to abort retry loop, False to continue retrying.
        """
        return False

    def _read_lines(self, out: StringIO | dict) -> pl.DataFrame:
        """
        Parse response data into Polars DataFrame.

        This method handles CSV data from StringIO buffers. For JSON format,
        subclasses should override this method.

        Parameters
        ----------
        out : StringIO or dict
            Data to parse. StringIO for CSV, dict for JSON.

        Returns
        -------
        pl.DataFrame
            Parsed data.

        Notes
        -----
        - Reverses row order to match pandas-datareader behavior
        - Strips whitespace from column names
        - Removes duplicate consecutive rows (Yahoo Finance quirk)
        - First column is treated as date/index column
        - Unlike pandas version, Polars has no index; first column is regular column
        """
        # Read CSV with Polars
        df = pl.read_csv(
            out,
            try_parse_dates=True,
            null_values=["-", "null"],
        )

        # Reverse row order (equivalent to pandas [::-1])
        df = df.reverse()

        # Strip whitespace from column names
        df = df.rename({col: col.strip() for col in df.columns})

        # Handle duplicate rows (Yahoo Finance issue)
        # Check if first column (date column) has duplicates at the end
        if len(df) > 2:
            first_col = df.columns[0]
            # Get last two values of first column
            last_two = df.select(first_col).tail(2)
            if len(last_two) == 2 and last_two[0, 0] == last_two[1, 0]:
                # Remove last row
                df = df.slice(0, len(df) - 1)

        # Note: Python 3.12+ handles unicode natively, no encoding needed
        # Polars column names are already strings

        return df

    def close(self) -> None:
        """Close HTTP client session and release resources."""
        self.session.close()

class DailyBaseReader(BaseReader):
    """
    Base reader for daily data with multi-symbol support.

    This class extends BaseReader to handle multiple symbols efficiently
    by processing them in chunks. It provides batch fetching capabilities
    and error resilience when some symbols fail.

    Parameters
    ----------
    symbols : str, int, Sequence[str], or pl.DataFrame
        Symbol(s) to fetch data for.
    start : str, date, datetime, int, or None, default None
        Start date for data retrieval.
    end : str, date, datetime, int, or None, default None
        End date for data retrieval.
    retry_count : int, default 3
        Number of retry attempts.
    pause : float, default 0.1
        Initial pause between retries.
    timeout : float, default 30.0
        Request timeout in seconds.
    session : httpx.Client or None, default None
        HTTP client session.
    freq : str or None, default None
        Data frequency.
    chunksize : int, default 25
        Number of symbols to process per batch.

    Attributes
    ----------
    chunksize : int
        Batch size for symbol processing.
    """

    def __init__(
        self,
        symbols: str | int | Sequence[str] | pl.DataFrame,
        start: str | datetime.date | datetime.datetime | int | None = None,
        end: str | datetime.date | datetime.datetime | int | None = None,
        retry_count: int = 3,
        pause: float = 0.1,
        timeout: float = 30.0,
        session: httpx.Client | None = None,
        freq: str | None = None,
        chunksize: int = 25,
    ) -> None:
        super().__init__(
            symbols,
            start,
            end,
            retry_count,
            pause,
            timeout,
            session,
            freq,
        )
        # FIX: Use public attribute, not private (was self._chunksize)
        self.chunksize = chunksize
    

    @abstractmethod
    def _get_params(self, symbol: str) -> dict[str, str | int | float]:
        """
        Get query parameters for a specific symbol.

        This method must be implemented by subclasses to provide
        symbol-specific query parameters.

        Parameters
        ----------
        symbol : str
            Symbol to get parameters for.

        Returns
        -------
        dict
            Query parameters for API request.

        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        """
        raise NotImplementedError("Subclass must implement '_get_params'")

    def read(self) -> pl.DataFrame:
        """
        Read data for one or more symbols.

        Returns
        -------
        pl.DataFrame
            Retrieved data. For multiple symbols, columns are organized
            in a flat structure with names like "{Attribute}_{Symbol}".

        Raises
        ------
        RemoteDataError
            If no data could be fetched for any symbol.
        """
        # If a single symbol (e.g., 'GOOG')
        if isinstance(self.symbols, (str, int)):
            df = self._read_one_data(self.url, params=self._get_params(self.symbols))
        # Or multiple symbols (e.g., ['GOOG', 'AAPL', 'MSFT'])
        # FIX: Use pl.DataFrame instead of undefined DataFrame
        elif isinstance(self.symbols, pl.DataFrame):
            # For DataFrame, use column names as symbols
            df = self._dl_mult_symbols(self.symbols.columns)
        else:
            df = self._dl_mult_symbols(self.symbols)
        return df

    def _dl_mult_symbols(self, symbols: Sequence[str]) -> pl.DataFrame:
        """
        Download data for multiple symbols with error handling.

        Parameters
        ----------
        symbols : Sequence[str]
            List of symbols to fetch.

        Returns
        -------
        pl.DataFrame
            Combined data for all symbols. Failed symbols are filled with null.
            Column names follow the pattern "{Attribute}_{Symbol}".

        Raises
        ------
        RemoteDataError
            If no symbols could be fetched successfully.

        Notes
        -----
        Symbols are processed in chunks of size `self.chunksize`. If a symbol
        fails to load, a warning is issued and the symbol is filled with nulls.
        Unlike pandas version which uses MultiIndex, this returns flat columns.
        """
        stocks: dict[str, pl.DataFrame] = {}
        failed: list[str] = []
        passed: list[str] = []

        for sym_group in _in_chunks(symbols, self.chunksize):
            for sym in sym_group:
                try:
                    stocks[sym] = self._read_one_data(
                        self.url, self._get_params(sym)
                    )
                    passed.append(sym)
                except (OSError, KeyError):
                    msg = "Failed to read symbol: {0!r}, replacing with null."
                    warnings.warn(msg.format(sym), SymbolWarning, stacklevel=2)
                    failed.append(sym)

        if len(passed) == 0:
            msg = "No data fetched using {0!r}"
            raise RemoteDataError(msg.format(self.__class__.__name__))

        try:
            # Handle case with both passed and failed symbols
            if len(stocks) > 0 and len(failed) > 0 and len(passed) > 0:
                # Create null template from first successful symbol
                df_na = stocks[passed[0]].clone()
                # Fill all data columns with null (keep structure)
                # Identify non-date columns (assume first column is date)
                data_cols = df_na.columns[1:] if len(df_na.columns) > 1 else []
                for col in data_cols:
                    df_na = df_na.with_columns(pl.lit(None).alias(col))

                # Add null dataframes for failed symbols
                for sym in failed:
                    stocks[sym] = df_na

            # Convert to wide format with symbols as columns
            # Strategy: Add symbol column to each df, concat, then pivot

            if len(stocks) == 1:
                # Single symbol - just return it
                return list(stocks.values())[0]

            # Multiple symbols - need to pivot
            dfs_with_symbol = []
            for symbol, df in stocks.items():
                # Add symbol identifier
                df = df.with_columns(pl.lit(symbol).alias("Symbol"))
                dfs_with_symbol.append(df)

            # Concat all dataframes vertically
            combined = pl.concat(dfs_with_symbol, how="vertical")

            # Get date column (assume first column)
            date_col = combined.columns[0]

            # Get value columns (all except first and Symbol)
            value_cols = [
                col
                for col in combined.columns
                if col not in [date_col, "Symbol"]
            ]

            # Pivot for each value column and join
            if len(value_cols) == 0:
                # No data columns, just return the combined dataframe
                return combined

            # Pivot first column to establish base
            result = combined.pivot(
                on="Symbol",
                index=date_col,
                values=value_cols[0],
            )

            # Rename columns to include attribute name
            # Format: {attribute}_{symbol}
            result = result.rename(
                {
                    col: f"{value_cols[0]}_{col}"
                    for col in result.columns
                    if col != date_col
                }
            )

            # Pivot and join remaining columns
            for value_col in value_cols[1:]:
                pivoted = combined.pivot(
                    on="Symbol",
                    index=date_col,
                    values=value_col,
                )
                # Rename columns
                pivoted = pivoted.rename(
                    {col: f"{value_col}_{col}" for col in pivoted.columns if col != date_col}
                )
                # Join with result
                result = result.join(pivoted, on=date_col, how="left")

            # Note: Column names format is "{Attribute}_{Symbol}"
            # This differs from pandas MultiIndex but provides similar information
            # Polars doesn't support hierarchical column names like pandas

            return result

        except AttributeError as exc:
            # Cannot construct result with just 1D nulls indicating no data
            msg = "No data fetched using {0!r}"
            raise RemoteDataError(msg.format(self.__class__.__name__)) from exc


def _in_chunks(seq: Sequence[str], size: int) -> Generator[Sequence[str], None, None]:
    """
    Split sequence into chunks of specified size.

    Parameters
    ----------
    seq : Sequence[str]
        Sequence to split.
    size : int
        Chunk size.

    Yields
    ------
    Sequence[str]
        Chunks of the sequence.

    Examples
    --------
    >>> list(_in_chunks(['A', 'B', 'C', 'D'], 2))
    [['A', 'B'], ['C', 'D']]
    """
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))




class OptionBaseReader(BaseReader):
    """
    Base reader for options data.

    This is a placeholder for future options data readers. All methods
    currently raise NotImplementedError and must be implemented by subclasses.

    Parameters
    ----------
    symbol : str
        Stock symbol for options.
    session : httpx.Client or None, default None
        HTTP client session.

    Attributes
    ----------
    symbol : str
        Stock symbol (uppercase).
    """

    def __init__(self, symbol: str, session: httpx.Client | None = None) -> None:
        """Instantiate options reader with a ticker symbol."""
        self.symbol = symbol.upper()
        super().__init__(symbols=symbol, session=session)

    def get_options_data(self, month=None, year=None, expiry=None):
        """
        ***Experimental***
        Gets call/put data for the stock with the expiration data in the
        given month and year
        """
        raise NotImplementedError

    def get_call_data(self, month=None, year=None, expiry=None):
        """
        ***Experimental***
        Gets call/put data for the stock with the expiration data in the
        given month and year
        """
        raise NotImplementedError

    def get_put_data(self, month=None, year=None, expiry=None):
        """
        ***Experimental***
        Gets put data for the stock with the expiration data in the
        given month and year
        """
        raise NotImplementedError

    def get_near_stock_price(
        self, above_below=2, call=True, put=False, month=None, year=None, expiry=None
    ):
        """
        ***Experimental***
        Returns a data frame of options that are near the current stock price.
        """
        raise NotImplementedError

    def get_forward_data(
        self, months, call=True, put=False, near=False, above_below=2
    ):  # pragma: no cover
        """
        ***Experimental***
        Gets either call, put, or both data for months starting in the current
        month and going out in the future a specified amount of time.
        """
        raise NotImplementedError

    def get_all_data(self, call=True, put=True):
        """
        ***Experimental***
        Gets either call, put, or both data for all available months starting
        in the current month.
        """
        raise NotImplementedError
