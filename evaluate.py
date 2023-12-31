from model.qg_fine_tune import QGFineTune
import argparse


def main(model_filepath):
    qg_fine_tune = QGFineTune(
        dataset_name="GEM/FairytaleQA",
        model_filepath=model_filepath,
    )
    test_dataset = qg_fine_tune.load_datasets(splits=["test"])[0]
    decoded_out_list, eval_scores = qg_fine_tune.batch_infer(test_dataset, batch_size=4)
    print(len(decoded_out_list))
    print(eval_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with QG model.")
    parser.add_argument("model_filepath", type=str, help="Filepath of the QG model.")
    args = parser.parse_args()
    main(args.model_filepath)
