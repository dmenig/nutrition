from process_nutrition_journal import main as process_nutrition_journal_main
from build_features import main as build_features_main
from train_model import main as train_model_main
from analyze_results import main as analyze_results_main

if __name__ == "__main__":
    print("Running process_nutrition_journal...")
    process_nutrition_journal_main()
    print("Finished process_nutrition_journal.")

    print("Running build_features...")
    build_features_main()
    print("Finished build_features.")

    print("Running train_model...")
    train_model_main()
    print("Finished train_model.")

    print("Running analyze_results...")
    analyze_results_main()
    print("Finished analyze_results.")

    print("Data pipeline finished.")
