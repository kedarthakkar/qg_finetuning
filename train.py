from model.qg_fine_tune import QGFineTune


def main():
    qg_fine_tune = QGFineTune(
        dataset_name='GEM/FairytaleQA',
        model_filepath="fairytale_qg_repo_test"
    )
    qg_fine_tune.train()


if __name__ == "__main__":
    main()
