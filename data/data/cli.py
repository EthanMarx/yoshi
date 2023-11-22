import logging
import os

from gwpy.timeseries import TimeSeriesDict
from jsonargparse import ActionConfigFile, ArgumentParser
from utils.logging import configure_logging

from data.find import DataQualityDict


def fetch(
    start: float, end: float, channels: list[str], sample_rate: float
) -> TimeSeriesDict:
    """
    Simple wrapper to annotate and simplify
    the kwargs so that jsonargparse can build
    a parser out of them.
    """

    logging.info(
        "Fetching {}s worth of data starting at GPS timestamp {}".format(
            end - start, start
        )
    )
    X = TimeSeriesDict.get(start=start, end=end, channels=channels)

    logging.info(f"Data downloaded, resampling to {sample_rate}Hz")
    return X.resample(sample_rate)


def main(args=None):
    query_parser = ArgumentParser()
    query_parser.add_method_arguments(DataQualityDict, "query_segments")
    query_parser.add_argument("--output-file", "-o", type=str)

    fetch_parser = ArgumentParser()
    fetch_parser.add_function_arguments(fetch)
    fetch_parser.add_argument("--sample-rate", type=float)
    fetch_parser.add_argument("--output-directory", "-o", type=str)
    fetch_parser.add_argument("--prefix", "-p", type=str, default="deepclean")

    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--verbose", type=bool, default=False)

    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("query", query_parser)
    subcommands.add_subcommand("fetch", fetch_parser)

    args = parser.parse_args(args)
    configure_logging(args.log_file, args.verbose)

    if args.subcommand == "query":
        args = args.query.as_dict()
        output_file = args.pop("output_file")

        logging.info(
            "Querying active segments in interval ({start}, {end})".format(
                **args
            )
        )
        segments = DataQualityDict.query_segments(**args)

        logging.info(
            "Discovered {} valid segments, writing to {}".format(
                len(segments), output_file
            )
        )
        segments.write(output_file)
    elif args.subcommand == "fetch":
        args = args.fetch.as_dict()
        output_directory = args.pop("output_directory")
        prefix = args.pop("prefix")
        X = fetch(**args)

        duration = args["end"] - args["start"]
        fname = "{}-{}-{}.hdf5".format(
            prefix, int(args["start"]), int(duration)
        )
        fname = os.path.join(output_directory, fname)

        logging.info(f"Writing downloaded data to {fname}")
        X.write(fname, format="hdf5")


if __name__ == "__main__":
    main()
