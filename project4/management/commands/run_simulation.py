"""Run the planning simulation for the project 4 user study."""

import time

from django.core.management.base import BaseCommand

from project4.simulation import RESULTS_PATH, run, save


class Command(BaseCommand):
    help = "Simulate both elicitation designs and write results/simulation.json."

    def add_arguments(self, parser):
        parser.add_argument("--participants", type=int, default=None)

    def handle(self, *args, **options):
        started = time.time()
        self.stdout.write("Simulating both designs. This takes a few minutes.")
        kwargs = {}
        if options["participants"]:
            kwargs["n_participants"] = options["participants"]
        results = run(**kwargs)
        save(results)
        power = results["power"]
        self.stdout.write(self.style.SUCCESS(
            "Done in %.0f s. Matched on time the ranking design is %+0.4f "
            "(d = %s, %s participants within); matched on decisions it is %+0.4f "
            "(%s within). Written to %s"
            % (time.time() - started, power["effect"], power["cohens_d"],
               power["within_total_80"], power["events_effect"],
               power["events_within_total_80"], RESULTS_PATH)))
