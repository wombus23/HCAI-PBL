"""Refit everything for project 3 and rewrite results/results.json."""

import time

from django.core.management.base import BaseCommand

from project3.experiments import RESULTS_PATH, run_all


class Command(BaseCommand):
    help = "Run the project 3 experiments and cache the results as JSON."

    def handle(self, *args, **options):
        started = time.time()
        self.stdout.write("Running the project 3 experiments. This takes a few minutes.")
        results = run_all()
        self.stdout.write(self.style.SUCCESS(
            "Done in %.0f s. Baseline %.4f, expert %.4f, team %.4f. Written to %s"
            % (time.time() - started,
               results["task1"]["accuracy"],
               results["task2"]["accuracy"],
               results["task3"]["learned"]["accuracy"],
               RESULTS_PATH)
        ))
