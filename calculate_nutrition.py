from nutrition_calculator import calculate_nutrient_from_formula


def main():
    """
    Main function to demonstrate on-the-fly nutrition calculation.
    """
    sample_formula = "100 * pain + 50 * fromage"
    nutrient_to_calculate = "Kcal"

    print(f"Calculating {nutrient_to_calculate} for formula: '{sample_formula}'")
    calculated_value = calculate_nutrient_from_formula(
        sample_formula, nutrient_to_calculate
    )

    print(f"Calculated {nutrient_to_calculate}: {calculated_value:.2f}")


if __name__ == "__main__":
    main()
