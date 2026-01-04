import json
import logging


# TODO: Use propper database
class CaloriesDatabase:
    def __init__(self, db_path):
        logging.info("Preparing database...")

        with open(db_path, "r") as file:
            self.db = json.load(file)

        logging.info("Database is ready.")

    def get_calories_per_100g(self, food_label):
        return self.db[food_label]["calories_per_100g"]

    def get_serving_size(self, food_label):
        return self.db[food_label]["serving_size"]
