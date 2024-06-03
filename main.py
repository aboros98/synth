import argparse
import concurrent
import multiprocessing
from typing import List, Optional, Tuple

from tqdm import tqdm

from synth.pipelines import AbstractPipeline, CodecLLMPipeline
from synth.utils.dataset_utils import load_dataset
from synth.utils.io_utils import load_yaml, save_results


def process_chunk(pipeline: AbstractPipeline,
                  dataset: List[str],
                  progress_queue: Optional[multiprocessing.Queue] = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Process a chunk of the dataset using the provided pipeline.

    Args:
        pipeline (AbstractPipeline): the pipeline to use
        dataset (List[str]): the dataset to process
        progress_queue (Optional[multiprocessing.Queue]): a queue to report progress

    Returns:
        Tuple[List[str], List[str], List[str]]: the generated dataset, the processed data, and the data that was not processed
    """
    generated_dataset, processed_data, not_processed_data = pipeline.run_pipeline(dataset, progress_queue)

    return generated_dataset, processed_data, not_processed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeCLLM pipeline")
    parser.add_argument("-c",
                        "--config_path",
                        type=str,
                        help="The path to the configuration file.",
                        required=True)
    parser.add_argument("-p",
                        "--num_parallel_processes",
                        type=int,
                        help="The number of parallel processes to use.",
                        required=True)

    args = parser.parse_args()

    config_path = args.config_path
    num_parallel_processes = args.num_parallel_processes

    # Load the configuration file
    config = load_yaml(config_path)

    # Initialize the pipeline
    pipeline = CodecLLMPipeline(config)

    # Load the dataset
    dataset = load_dataset(config["pipeline"]["dataset_path"])

    chunk_size = len(dataset) // num_parallel_processes
    chunks = [dataset[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)]

    with multiprocessing.Manager() as manager:
        progress_queue = manager.Queue()
        processed_data_counter = manager.Value('i', 0)
        lock = manager.Lock()

        total_data_points = sum(len(chunk) for chunk in chunks)
        with tqdm(total=total_data_points, desc="a/chemy(ing) 🪄 ...") as progress_bar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel_processes) as executor:
                futures = [
                    executor.submit(
                        process_chunk, pipeline, chunk, progress_queue)
                    for chunk in chunks
                ]

                results = []
                while any(future.running() for future in futures):
                    try:
                        processed_count = progress_queue.get(timeout=1)
                        progress_bar.update(processed_count)
                    except Exception:
                        pass

                for future in futures:
                    results.append(future.result())

    generated_dataset = []
    processed_data = []
    skipped_data = []

    for result in results:
        generated_dataset.extend(result[0])
        processed_data.extend(result[1])
        skipped_data.extend(result[2])

    save_results(generated_dataset=generated_dataset,
                 processed_data=processed_data,
                 skipped_data=skipped_data,
                 output_path=config["pipeline"]["output_path"])
