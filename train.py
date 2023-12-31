from model.qg_fine_tune import Seq2SeqFineTune


def main():
    qg_fine_tune = Seq2SeqFineTune(
        dataset_name="GEM/FairytaleQA", model_filepath="fairytale_qg_repo_test"
    )
    qg_fine_tune.train()


if __name__ == "__main__":
    main()
