from app.db.populate_db import populate_sport_activities_table, verify_population


def main() -> None:
    populate_sport_activities_table()
    verify_population()


if __name__ == "__main__":
    main()


